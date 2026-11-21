"""Bounded sizing/overfit for ONE M4 model, adapted from M3 probe cb135234.

No main training follows the probe. All probe-created .pt files are deleted.
"""

import argparse
import copy
import fcntl
import gc
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from ..m2.download import sha256
from ..m2.preflight import verify_locks
from ..m3.common import seed_all, write_json
from ..m3.data import Store, to_device
from ..m3.engine import backward_microbatch, diagnostics, save_checkpoint
from ..m3.probe import choose_batch, positive_case, select_cases, stress_window
from .common import GPU_LOCK, PROJECT, config, identity, require_committed, verify_parent, verify_self
from .data import collate
from .models import RelationModel
from .preflight import check_features, check_runtime


def measure(kind, cfg, samples, size, device):
    seed_all(17)
    model = optimizer = batch = None
    times = []
    try:
        model = RelationModel(kind, cfg).to(device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
        batch = to_device(collate([samples[i % len(samples)] for i in range(size)]), device)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
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
            parameters=sum(p.numel() for p in model.parameters()),
        )
    finally:
        del optimizer, model, batch
        gc.collect()
        torch.cuda.empty_cache()


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
    parser.add_argument("--model", choices=tuple(config()["models"]), default="pvsg_conv_k5_l3")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    os.chdir(PROJECT)
    require_committed()
    verify_locks()
    verify_parent()
    verify_self()
    check_runtime()
    cfg = config()
    args.output = args.output or Path(cfg["output_root"]) / "profiles" / f"{args.model}.json"
    feature_sha = check_features(cfg)  # Refuses to initialize CUDA while M2 is incomplete/running.
    if args.output.exists():
        raise ValueError("Profile already exists; preserve it and use --output for a new profile version")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Path(cfg["output_root"]).mkdir(parents=True, exist_ok=True)
    GPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with GPU_LOCK.open("a") as lock:
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
            identity=identity(cfg, feature_sha),
            selected_model=args.model,
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
        with tempfile.TemporaryDirectory(prefix="pvsg_m4_probe_", dir=args.output.parent) as temporary:
            for kind in [args.model]:
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
