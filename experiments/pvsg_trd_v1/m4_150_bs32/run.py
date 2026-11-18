"""Single M4 run or sizing probe sharing a GPU with M3."""

import argparse
import fcntl
import os
import tempfile
from pathlib import Path

import torch

from ..m2.download import sha256
from ..m2.preflight import verify_locks
from ..m3.common import write_json
from ..m3.data import Store
from ..m3.preflight import check_features, check_runtime
from ..m3.probe import positive_case, select_cases, stress_window
from ..m4.common import PROJECT, config, identity, require_committed, verify_parent, verify_self
from ..m4.preflight import check_profile
from ..m4.probe import measure, overfit
from ..m4.runner import train


def settings():
    return dict(
        config(),
        max_epochs=150,
        early_stopping_patience=15,
        experiment_id="pvsg-gt-trd-m4-150-bs32-v1",
        output_root="reports/pvsg-trd-m4-150-bs32-v1",
        allocator_fraction=0.60,
        launcher_sha256=sha256(__file__),
    )


def profile(cfg, kind, digest, device, path, budget):
    store = Store(cfg["dataset_cache"], cfg["feature_cache"], cache_size=6)
    _, cases = select_cases(store, cfg["window_frames"])
    samples = [store.window(r["video_id"], r["start"], cfg["window_frames"]) for r in cases]
    heavy = stress_window(cfg["window_frames"])
    size = cfg["effective_batch_windows"]
    peak = measure(kind, cfg, [heavy], size, device)
    mixed = measure(kind, cfg, samples, size, device)
    if max(peak["peak_reserved_bytes"], mixed["peak_reserved_bytes"]) > budget:
        raise ValueError("Batch 32 exceeds current GPU budget; retry with more free VRAM")
    results = [dict(state="passed", batch_size=size, heavy=peak, mixed=mixed)]
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
    # Serialize both M4 launchers while allowing the user's M3 queue to proceed.
    lock_path = Path("reports/pvsg-trd-m4-150-v1/.gpu.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        device = torch.device("cuda:0")
        torch.cuda.set_per_process_memory_fraction(cfg["allocator_fraction"], device)
        if not torch.cuda.is_bf16_supported():
            raise ValueError("BF16 GPU required")
        free, total = torch.cuda.mem_get_info(device)
        budget = int(min(free - 2**30, total * cfg["allocator_fraction"] * 0.97))
        if args.action == "profile":
            if path.exists():
                check_profile(path, cfg, digest, args.model)
                return
            profile(cfg, args.model, digest, device, path, budget)
            return
        report = check_profile(path, cfg, digest, args.model)
        selected = report["models"][args.model]
        row = next(r for r in selected["candidates"] if r["batch_size"] == selected["batch_size"])
        if selected["batch_size"] != cfg["effective_batch_windows"]:
            raise ValueError("Physical batch must equal effective batch")
        if (
            report["gpu"]["name"] != torch.cuda.get_device_name(device)
            or max(row[k]["peak_reserved_bytes"] for k in ("mixed", "heavy")) > budget
        ):
            raise ValueError("Insufficient profiled GPU headroom or different device")
        train(cfg, args.model, args.seed, report, digest, device)


if __name__ == "__main__":
    main()
