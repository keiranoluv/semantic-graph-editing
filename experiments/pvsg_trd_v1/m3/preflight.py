"""CPU-only gates: immutable earlier stages, complete cache, tested batch profile."""

import argparse
import fcntl
from pathlib import Path

from ..contracts import read_json
from ..m2.download import sha256
from ..m2.features import cache_identity
from ..m2.preflight import verify_locks
from ..validate_m0 import canonical_sha256
from .common import ROOT, code_identity, config, require_committed, runtime_versions


def check_runtime():
    if runtime_versions() != read_json(ROOT / "runtime.json"):
        raise ValueError("M3 runtime mismatch")


def check_features(cfg):
    root = Path(cfg["feature_cache"])
    if not (root / "summary.json").is_file():
        raise ValueError("M2 extraction is incomplete: run bash experiments/pvsg_trd_v1/m2/launch_extract.sh first")
    with (root / ".lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("M2 extractor is still running; do not compete for its GPU") from error
    summary = read_json(root / "summary.json")
    manifest = read_json(ROOT.parent / "m2/manifests/dataset.lock.json")
    expected = set(manifest["video_sources"])
    if (
        not summary["complete_cohort"]
        or summary["videos"] != 400
        or len(summary["video_ids"]) != 400
        or set(summary["video_ids"]) != expected
        or set(summary["feature_hashes"]) != expected
        or summary["rows"] != 1512183
        or summary["visual_rows"] != 1512134
        or summary["training"]
    ):
        raise ValueError("M2 feature summary cohort/row counts mismatch")
    feature_cfg = read_json(ROOT.parent / "m2/configs/clip_masked_v1.json")
    if summary["feature_id"] != feature_cfg["feature_id"]:
        raise ValueError("Feature variant mismatch")
    actual = summary["identity"]
    if actual != cache_identity(feature_cfg, actual["device_type"], actual["batch_size"]):
        raise ValueError("M2 feature extraction identity changed")
    for vid in sorted(expected):
        meta = read_json(root / f"{vid}.json")
        if (
            sha256(root / f"{vid}.npz") != summary["feature_hashes"][vid]
            or meta["feature_sha256"] != summary["feature_hashes"][vid]
            or meta["identity"] != actual
        ):
            raise ValueError(f"Feature shard corrupt/incompatible: {vid}")
    return sha256(root / "summary.json")


def check_profile(path, cfg, feature_sha):
    if not Path(path).is_file():
        raise ValueError(
            "GPU profile missing: run launch_profile.sh after M2 and review sizing/overfit before training"
        )
    profile = read_json(path)
    if (
        profile["state"] != "passed"
        or profile["code_sha256"] != code_identity()
        or profile["config_sha256"] != canonical_sha256(cfg)
        or profile["feature_summary_sha256"] != feature_sha
        or profile["runtime"] != runtime_versions()
    ):
        raise ValueError("A successful GPU profile for this exact code/config/cache is required")
    for model in cfg["models"]:
        result = profile["models"][model]
        if not result["overfit"]["passed"] or result["batch_size"] not in cfg["profile_batch_candidates"]:
            raise ValueError(f"Unverified batch/convergence smoke check: {model}")
        if cfg["effective_batch_windows"] % result["batch_size"]:
            raise ValueError("Physical batch must divide the effective batch")
    return profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-only", action="store_true", help="Check code/runtime while M2 is running; no GPU")
    parser.add_argument("--profile", type=Path, default=Path("reports/pvsg-trd-m3-v1/gpu_profile.json"))
    args = parser.parse_args()
    verify_locks()
    check_runtime()
    if args.cpu_only:
        print("CPU locks/runtime passed; full features and GPU sizing not asserted")
        return
    require_committed()
    cfg = config()
    digest = check_features(cfg)
    check_profile(args.profile, cfg, digest)
    print("Main-training preflight passed; no GPU work was started")


if __name__ == "__main__":
    main()
