"""Balanced four-window throughput calibration and a fixed 400-step real-fit diagnostic."""

import fcntl
import sys
import tempfile
from pathlib import Path

import torch

from ..contracts import read_json
from ..m2.download import sha256
from ..m2.preflight import verify_locks
from ..m3.common import PROJECT, code_identity, config, require_committed, runtime_versions, write_json
from ..m3.data import Store
from ..m3.preflight import check_features, check_runtime
from ..m3.probe import choose_batch, measure, overfit, positive_case, select_cases
from ..validate_m0 import canonical_sha256


def balanced_measure(kind, cfg, cases, size, device):
    # Every size sees all four density quantiles with equal window weights.
    parts = [measure(kind, cfg, cases[i:] + cases[:i], size, device) for i in range(0, len(cases), size)]
    seconds = sum(row["seconds_mean"] for row in parts) / len(parts)
    return dict(
        batch_size=size,
        windows_per_second=size / seconds,
        seconds_mean=seconds,
        peak_reserved_bytes=max(row["peak_reserved_bytes"] for row in parts),
        measurements=parts,
    )


def main():
    cfg = config()
    require_committed()
    verify_locks()
    check_runtime()
    digest = check_features(cfg)
    root = PROJECT / cfg["output_root"]
    parent = root / "gpu_profile.json"
    prior = read_json(parent)
    for key, expected in (
        ("feature_summary_sha256", digest),
        ("code_sha256", code_identity()),
        ("config_sha256", canonical_sha256(cfg)),
        ("runtime", runtime_versions()),
    ):
        if prior[key] != expected:
            raise ValueError(f"Incompatible sizing evidence: {key}")
    out = root / "gpu_profile_balanced_v2.json"
    with (root / ".gpu.lock").open("a") as lock, tempfile.TemporaryDirectory(dir=root) as tmp:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if out.exists():
            raise FileExistsError(out)
        device = torch.device("cuda:0")
        if torch.cuda.get_device_name(device) != prior["gpu"]["name"]:
            raise ValueError("GPU changed since heavy-window measurement")
        free, total = torch.cuda.mem_get_info(device)
        budget = min(int(total * cfg["profile_memory_fraction"]), int(free * 0.9))
        store = Store(cfg["dataset_cache"], cfg["feature_cache"], cache_size=6)
        _, mixed = select_cases(store, cfg["window_frames"])
        if mixed != prior["mixed_cases"]:
            raise ValueError("Mixed cases changed")
        cases = [store.window(r["video_id"], r["start"], cfg["window_frames"]) for r in mixed]
        tiny = positive_case(store)
        report = dict(
            prior,
            state="running",
            models={},
            parent_profile_sha256=sha256(parent),
            calibration_code_sha256=sha256(__file__),
            calibration_steps=400,
            method="Equal weight for four fit-density cases at every BS. Heavy evidence from parent profile.",
        )
        report["gpu"] = dict(prior["gpu"], free_before_probe_bytes=free, reserved_memory_budget_bytes=budget)
        del report["test_checkpoints_remaining"]
        for kind in cfg["models"]:
            results = []
            for row in prior["models"][kind]["candidates"]:
                if row["state"] != "passed":
                    continue
                size = row["batch_size"]
                measured = balanced_measure(kind, cfg, cases, size, device)
                results.append(dict(state="passed", batch_size=size, heavy=row["heavy"], mixed=measured))
            size = choose_batch(results, budget)
            convergence = overfit(kind, cfg, tiny, device, tmp, steps=400)
            report["models"][kind] = dict(
                batch_size=size, accumulation_steps=32 // size, candidates=results, overfit=convergence
            )
            print(kind, size, convergence["passed"], flush=True)
        report["test_checkpoints_remaining"] = len(list(Path(tmp).rglob("*.pt")))
        passed = all(r["overfit"]["passed"] for r in report["models"].values())
        report["state"] = "passed" if passed and not report["test_checkpoints_remaining"] else "failed_overfit"
        write_json(out, report)
        return int(report["state"] != "passed")


if __name__ == "__main__":
    sys.exit(main())
