"""M3 schedule with 150 epochs and early stopping patience 15."""

import argparse
import fcntl
import os
from pathlib import Path

import torch

from extension_pvsg.m2.download import sha256
from extension_pvsg.m2.preflight import verify_locks
from extension_pvsg.m3.common import PROJECT, config, require_committed
from extension_pvsg.m3.preflight import check_features, check_profile, check_runtime
from extension_pvsg.m3.runner import train
from extension_pvsg.m3_150.summary import summarize


def setup():
    require_committed()
    verify_locks()
    check_runtime()
    base = config()
    digest = check_features(base)
    previous = Path(base["output_root"])
    profile = check_profile(previous / "gpu_profile_balanced_v2.json", base, digest)
    # The duration changes, but each optimizer update uses the profiled M3 implementation.
    cfg = dict(
        base,
        max_epochs=150,
        early_stopping_patience=15,
        experiment_id="pvsg-gt-trd-m3-150-v1",
        output_root="reports/pvsg-trd-m3-150-v1",
        launcher_sha256=sha256(__file__),
    )
    return cfg, digest, profile, previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate cache, runtime and sizing on CPU")
    args = parser.parse_args()
    os.chdir(PROJECT)
    cfg, digest, profile, previous = setup()
    for model in cfg["models"]:
        print(model, "physical batch", profile["models"][model]["batch_size"], flush=True)
    if args.check:
        return
    with (previous / ".queue.lock").open("a") as queue, (previous / ".gpu.lock").open("a") as gpu:
        for lock in (queue, gpu):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not torch.cuda.is_bf16_supported() or torch.cuda.get_device_name(0) != profile["gpu"]["name"]:
            raise ValueError("GPU differs from the BF16 sizing profile")
        for model in cfg["models"]:
            selected = profile["models"][model]
            row = next(r for r in selected["candidates"] if r["batch_size"] == selected["batch_size"])
            required = max(row[case]["peak_reserved_bytes"] for case in ("heavy", "mixed"))
            for seed in cfg["seeds"]:
                if torch.cuda.mem_get_info(0)[0] < required * 1.1:
                    raise ValueError("Insufficient free VRAM for the profiled batch")
                train(cfg, model, seed, profile, digest, torch.device("cuda:0"))
                torch.cuda.empty_cache()
        summarize(cfg)


if __name__ == "__main__":
    main()
