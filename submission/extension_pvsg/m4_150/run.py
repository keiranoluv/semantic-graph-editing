"""Single M4 run or sizing probe sharing a GPU with M3."""

import argparse
import fcntl
import os
import tempfile
from pathlib import Path

import torch

from extension_pvsg.m2.download import sha256
from extension_pvsg.m2.preflight import verify_locks
from extension_pvsg.m3.common import write_json
from extension_pvsg.m3.data import Store
from extension_pvsg.m3.preflight import check_features, check_runtime
from extension_pvsg.m3.probe import choose_batch, positive_case, select_cases, stress_window
from extension_pvsg.m4.common import PROJECT, config, identity, require_committed, verify_parent, verify_self
from extension_pvsg.m4.preflight import check_profile
from extension_pvsg.m4.probe import measure, overfit
from extension_pvsg.m4.runner import train


def settings():
    return dict(
        config(),
        max_epochs=150,
        early_stopping_patience=15,
        experiment_id="pvsg-gt-trd-m4-150-v1",
        output_root="reports/pvsg-trd-m4-150-v1",
        allocator_fraction=0.45,
        loader_workers=2,
        prefetch_factor=1,
        launcher_sha256=sha256(__file__),
    )


def profile(cfg, kind, digest, device, path, budget):
    store = Store(cfg["dataset_cache"], cfg["feature_cache"], cache_size=6)
    _, cases = select_cases(store, cfg["window_frames"])
    samples = [store.window(r["video_id"], r["start"], cfg["window_frames"]) for r in cases]
    heavy = stress_window(cfg["window_frames"])
    results = []
    for size in cfg["profile_batch_candidates"]:
        try:
            peak = measure(kind, cfg, [heavy], size, device)
            if peak["peak_reserved_bytes"] > budget:
                break
            timings = [
                measure(kind, cfg, samples[i:] + samples[:i], size, device) for i in range(0, len(samples), size)
            ]
            seconds = sum(r["seconds_mean"] for r in timings) / len(timings)
            mixed = dict(
                peak_reserved_bytes=max(r["peak_reserved_bytes"] for r in timings),
                windows_per_second=size / seconds,
                measurements=timings,
            )
            results.append(dict(state="passed", batch_size=size, heavy=peak, mixed=mixed))
            print(f"batch={size} mixed={size/seconds:.1f} windows/s", flush=True)
        except torch.cuda.OutOfMemoryError:
            break
    torch.cuda.empty_cache()
    size = choose_batch(results, budget)
    with tempfile.TemporaryDirectory() as tmp:
        smoke = overfit(kind, cfg, positive_case(store), device, tmp, steps=400)
        remaining = len(list(Path(tmp).rglob("*.pt")))
    report = dict(
        state="passed" if smoke["passed"] and not remaining else "failed_overfit",
        models={kind: dict(batch_size=size, candidates=results, overfit=smoke)},
        selected_model=kind,
        identity=identity(cfg, digest),
        mixed_cases=cases,
        gpu=dict(name=torch.cuda.get_device_name(device), reserved_memory_budget_bytes=budget),
        test_checkpoints_remaining=remaining,
    )
    write_json(path, report)
    check_profile(path, cfg, digest, kind)


def main():
    cfg = settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["check", "profile", "train"])
    parser.add_argument("--model", choices=cfg["models"], default="pvsg_conv_k5_l3")
    parser.add_argument("--seed", type=int, choices=cfg["seeds"], default=17)
    args = parser.parse_args()
    os.chdir(PROJECT)
    require_committed()
    verify_parent()
    verify_self()
    verify_locks()
    check_runtime()
    digest = check_features(cfg)
    if args.action == "check":
        print("CPU checks passed", cfg["max_epochs"], cfg["early_stopping_patience"])
        return
    root = Path(cfg["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    path = root / "profiles" / f"{args.model}.json"
    with (root / ".gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        device = torch.device("cuda:0")
        torch.cuda.set_per_process_memory_fraction(cfg["allocator_fraction"], device)
        if not torch.cuda.is_bf16_supported():
            raise ValueError("BF16 GPU required")
        free, total = torch.cuda.mem_get_info(device)
        budget = int(min(free * 0.85, total * cfg["allocator_fraction"] * 0.9))
        if args.action == "profile":
            if path.exists():
                check_profile(path, cfg, digest, args.model)
                return
            profile(cfg, args.model, digest, device, path, budget)
            return
        report = check_profile(path, cfg, digest, args.model)
        selected = report["models"][args.model]
        row = next(r for r in selected["candidates"] if r["batch_size"] == selected["batch_size"])
        if (
            report["gpu"]["name"] != torch.cuda.get_device_name(device)
            or max(row[k]["peak_reserved_bytes"] for k in ("mixed", "heavy")) > budget
        ):
            raise ValueError("Insufficient profiled GPU headroom or different device")
        train(cfg, args.model, args.seed, report, digest, device)


if __name__ == "__main__":
    main()
