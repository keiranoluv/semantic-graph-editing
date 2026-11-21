"""CPU-only provenance/cache/profile gates for a selected M4 model."""

import argparse
import os
from pathlib import Path

from extension_pvsg.contracts import read_json
from extension_pvsg.m2.preflight import verify_locks
from extension_pvsg.m3.preflight import check_features, check_runtime
from extension_pvsg.m4.common import PROJECT, config, identity, require_committed, verify_parent, verify_self


def check_profile(path, cfg, feature_sha, kind):
    if not Path(path).is_file():
        raise ValueError(f"Profile {kind} first with M4 launch_profile.sh; M3 batch sizing cannot be reused")
    profile = read_json(path)
    if (
        profile.get("state") != "passed"
        or profile.get("identity") != identity(cfg, feature_sha)
        or profile.get("selected_model") != kind
        or set(profile.get("models", {})) != {kind}
        or profile.get("test_checkpoints_remaining") != 0
    ):
        raise ValueError("Require a passed M4 profile for exactly this model/code/config/cache, with .pt cleanup")
    result = profile["models"][kind]
    size = result["batch_size"]
    if not result["overfit"]["passed"] or size not in cfg["profile_batch_candidates"]:
        raise ValueError("Batch size or positive-overfit check is unverified")
    if cfg["effective_batch_windows"] % size:
        raise ValueError("Physical batch must divide effective batch")
    selected = [row for row in result["candidates"] if row["batch_size"] == size and row["state"] == "passed"]
    if len(selected) != 1:
        raise ValueError("Selected batch lacks unique measured evidence")
    budget = profile["gpu"]["reserved_memory_budget_bytes"]
    if any(selected[0][case]["peak_reserved_bytes"] > budget for case in ("heavy", "mixed")):
        raise ValueError("Selected batch exceeds reserved VRAM budget")
    return profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--model", choices=tuple(config()["models"]), default="pvsg_conv_k5_l3")
    parser.add_argument("--profile", type=Path)
    args = parser.parse_args()
    os.chdir(PROJECT)
    verify_locks()
    verify_parent()
    verify_self()
    check_runtime()
    cfg = config()
    if args.cpu_only:
        print("M0--M4 locks/runtime passed on CPU; no cache completeness or GPU sizing asserted")
        return
    require_committed()
    digest = check_features(cfg)
    path = args.profile or Path(cfg["output_root"]) / "profiles" / f"{args.model}.json"
    result = check_profile(path, cfg, digest, args.model)
    print(f"Ready: {args.model}, physical BS={result['models'][args.model]['batch_size']}, effective BS=32")


if __name__ == "__main__":
    main()
