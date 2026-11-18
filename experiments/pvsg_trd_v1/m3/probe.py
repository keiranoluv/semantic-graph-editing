"""Bounded GPU sizing and tiny-fit checks; no main training and no .pt retained."""

import argparse
import copy
import fcntl
import gc
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from ..m2.download import sha256
from ..m2.preflight import verify_locks
from ..validate_m0 import canonical_sha256
from .common import code_identity, config, require_committed, runtime_versions, seed_all, write_json
from .data import Store, collate, make_window, to_device
from .engine import backward_microbatch, diagnostics, save_checkpoint
from .models import RelationModel
from .preflight import check_features, check_runtime


def select_cases(store, size):
    rows = []
    for vid in sorted(store.ids("fit")):
        with np.load(store.cache_root / "geometry" / f"{vid}.npz", allow_pickle=False) as z:
            counts = z["visible"].sum(axis=0)
        q = counts * (counts - 1)
        width = min(size, len(q))
        sums = np.r_[0, q.cumsum()]
        density = sums[width:] - sums[:-width]
        start = int(np.argmax(density))
        rows.append(dict(video_id=vid, start=start, queries=int(density[start])))
    rows.sort(key=lambda r: (r["queries"], r["video_id"], r["start"]))
    mixed = [rows[round((len(rows) - 1) * p)] for p in (0.25, 0.5, 0.75, 0.95)]
    return rows[-1], mixed


def measure(kind, cfg, samples, size, device):
    seed_all(17)
    model = RelationModel(kind, cfg).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    batch = to_device(collate([samples[i % len(samples)] for i in range(size)]), device)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    times = []
    try:
        for step in range(cfg["profile_warmup_steps"] + cfg["profile_measured_steps"]):
            started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            backward_microbatch(
                model, batch, chunk_size=cfg["pair_chunk_size"], group_windows=size, precision=cfg["precision"]
            )
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_norm"], error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize(device)
            if step >= cfg["profile_warmup_steps"]:
                times.append(time.monotonic() - started)
        return dict(
            batch_size=size,
            seconds_mean=float(np.mean(times)),
            seconds_std=float(np.std(times)),
            windows_per_second=size / float(np.mean(times)),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
            nodes=batch["nodes"].shape[0],
            query_frames=batch["query_edges"].shape[1],
            state="passed",
        )
    finally:
        del optimizer, model, batch
        gc.collect()
        torch.cuda.empty_cache()


def choose_batch(results, memory_budget):
    valid = [
        r
        for r in results
        if r["state"] == "passed"
        and r["heavy"]["peak_reserved_bytes"] <= memory_budget
        and r["mixed"]["peak_reserved_bytes"] <= memory_budget
    ]
    if not valid:
        raise ValueError("No batch passed the reserved-VRAM headroom constraint")
    fastest = max(r["mixed"]["windows_per_second"] for r in valid)
    # Avoid spending memory on a larger batch within timing noise of a smaller one.
    return min(r["batch_size"] for r in valid if r["mixed"]["windows_per_second"] >= 0.95 * fastest)


def stress_window(size, objects=38):
    """Synthetic upper envelope from M2's maximum simultaneous visible count.

    All 38 objects visible for every frame is stricter than any audited window;
    no test relation labels, measured accuracy or fitted data priors are used.
    """
    rng = np.random.default_rng(17)
    ids = np.arange(1, objects + 1)
    geom = np.zeros((objects, size, 6), dtype=np.float32)
    geom[:, :, 0] = ((np.arange(objects) % 8 + 0.5) / 8)[:, None]
    geom[:, :, 1] = ((np.arange(objects) // 8 + 0.5) / 5)[:, None]
    geom[:, :, 2:] = [0.1, 0.15, 0.015, 0.01]
    geometry = dict(
        track_ids=ids,
        class_ids=np.arange(objects),
        geometry=geom,
        visible=np.ones((objects, size), dtype=bool),
        motion=np.zeros((objects, size, 4), dtype=np.float32),
        delta_frames=np.ones((objects, size), dtype=np.int64),
        rgb_available=np.ones(size, dtype=bool),
    )
    features = rng.normal(size=(objects * size, 512)).astype(np.float32)
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    visual = dict(
        features=features,
        frame_ids=np.repeat(np.arange(size), objects),
        track_ids=np.tile(ids, size),
        visual_available=np.ones(objects * size, dtype=bool),
    )
    video = dict(video_id="synthetic_m3_vram_envelope", num_frames=size)
    return make_window(video, geometry, visual, 0, size, target={"relations": []})


def positive_case(store, size=16):
    candidates = []
    for vid in sorted(store.ids("fit")):
        for row in store.targets[vid]["relations"]:
            for start, stop in row["intervals"]:
                candidates.append((stop - start + 1, vid, start))
    for _, vid, start in sorted(candidates, reverse=True):
        start = min(start, max(0, store.videos[vid]["num_frames"] - size))
        sample = store.window(vid, start, size)
        # A bounded case makes overfitting a pipeline test, not a second training run.
        if 2 <= sample["graph"].query_edges.shape[1] <= 4096 and sample["labels"].sum() > 0:
            return sample
    raise ValueError("No bounded fit window with positive supervision")


def overfit(kind, cfg, sample, device, checkpoint_dir, steps=200):
    seed_all(17)
    probe_cfg = copy.deepcopy(cfg)
    probe_cfg["dropout"] = 0.0
    model = RelationModel(kind, probe_cfg).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=0)
    batch = to_device(collate([sample]), device)
    first = last = None
    started = time.monotonic()
    path = Path(checkpoint_dir) / f"{kind}_probe.pt"
    try:
        for _step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            totals = backward_microbatch(
                model,
                batch,
                chunk_size=cfg["pair_chunk_size"],
                group_windows=1,
                precision=cfg["precision"] if device.type == "cuda" else "fp32",
            )
            stats = diagnostics(totals, 1)
            if first is None:
                first = stats
            last = stats
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_norm"], error_if_nonfinite=True)
            optimizer.step()
        save_checkpoint(path, dict(model=model.state_dict(), kind=kind))
        restored = torch.load(path, map_location="cpu", weights_only=True)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value.cpu(), restored["model"][name], rtol=0, atol=0)
        checksum = sha256(path)
        return dict(
            passed=last["loss"] < 0.7 * first["loss"]
            and last["positive_bce"] < first["positive_bce"]
            and last["frame_recall"] > 0,
            first=first,
            last=last,
            steps=steps,
            seconds=time.monotonic() - started,
            video_id=sample["graph"].video_id,
            start=sample["start"],
            stop=sample["stop"],
            dropout=0.0,
            weight_decay=0.0,
            note="Memorization smoke check on one fit window; not validation/generalization",
            checkpoint_roundtrip_sha256=checksum,
            test_checkpoint_deleted=True,
        )
    finally:
        path.unlink(missing_ok=True)
        path.with_suffix(".partial.pt").unlink(missing_ok=True)
        del model, optimizer, batch
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/pvsg-trd-m3-v1/gpu_profile.json"))
    args = parser.parse_args()
    require_committed()
    verify_locks()
    check_runtime()
    cfg = config()
    feature_sha = check_features(cfg)  # Refuses to initialize CUDA while M2 is incomplete/running.
    if args.output.exists():
        raise ValueError("Profile already exists; preserve it and use --output for a new profile version")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Path(cfg["output_root"]).mkdir(parents=True, exist_ok=True)
    with (Path(cfg["output_root"]) / ".gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise ValueError("CUDA with BF16 is required for this profile")
        device = torch.device("cuda:0")
        free, total = torch.cuda.mem_get_info(device)
        budget = min(int(total * cfg["profile_memory_fraction"]), int(free * 0.9))
        store = Store(cfg["dataset_cache"], cfg["feature_cache"], cache_size=6)
        heavy, mixed = select_cases(store, cfg["window_frames"])
        heavy_sample = stress_window(cfg["window_frames"])
        mixed_samples = [store.window(r["video_id"], r["start"], cfg["window_frames"]) for r in mixed]
        tiny = positive_case(store)
        report = dict(
            state="running",
            code_sha256=code_identity(),
            config_sha256=canonical_sha256(cfg),
            feature_summary_sha256=feature_sha,
            runtime=runtime_versions(),
            gpu=dict(
                name=torch.cuda.get_device_name(device),
                total_bytes=total,
                free_before_probe_bytes=free,
                reserved_memory_budget_bytes=budget,
            ),
            densest_fit_case=heavy,
            heavy_case={
                "kind": "synthetic upper envelope",
                "visible_objects": 38,
                "frames": cfg["window_frames"],
                "nodes": len(heavy_sample["graph"].nodes),
                "query_frames": heavy_sample["graph"].query_edges.shape[1],
            },
            mixed_cases=mixed,
            method="Resident-tensor forward/backward/AdamW timing; CPU loader throughput is reported by main epochs",
            automatic_main_training=False,
            models={},
        )
        write_json(args.output, report)
        with tempfile.TemporaryDirectory(prefix="pvsg_m3_probe_", dir=args.output.parent) as temporary:
            for kind in cfg["models"]:
                results = []
                for size in cfg["profile_batch_candidates"]:
                    try:
                        heavy_result = measure(kind, cfg, [heavy_sample], size, device)
                        mixed_result = measure(kind, cfg, mixed_samples, size, device)
                        results.append(dict(state="passed", batch_size=size, heavy=heavy_result, mixed=mixed_result))
                        print(
                            f"{kind} batch={size}: {mixed_result['windows_per_second']:.2f} windows/s, worst reserved={heavy_result['peak_reserved_bytes']/2**30:.2f} GiB",
                            flush=True,
                        )
                        if heavy_result["peak_reserved_bytes"] > budget:
                            break
                    except torch.cuda.OutOfMemoryError:
                        gc.collect()
                        torch.cuda.empty_cache()
                        results.append(dict(state="oom", batch_size=size))
                        break
                size = choose_batch(results, budget)
                convergence = overfit(kind, cfg, tiny, device, temporary)
                report["models"][kind] = dict(
                    batch_size=size,
                    accumulation_steps=cfg["effective_batch_windows"] // size,
                    candidates=results,
                    overfit=convergence,
                )
                write_json(args.output, report)
            report["state"] = (
                "passed" if all(r["overfit"]["passed"] for r in report["models"].values()) else "failed_overfit"
            )
            report["test_checkpoints_remaining"] = len(list(Path(temporary).rglob("*.pt")))
            if report["test_checkpoints_remaining"]:
                raise ValueError("Probe cleanup incomplete")
        write_json(args.output, report)
        print(f"Profile: {report['state']}; no test .pt kept; no main training launched", flush=True)
        if report["state"] != "passed":
            raise SystemExit(2)


if __name__ == "__main__":
    main()
