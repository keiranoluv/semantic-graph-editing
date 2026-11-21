"""CPU-only gates: immutable earlier stages, complete cache, tested batch profile."""

import fcntl
from pathlib import Path

from extension_pvsg.contracts import read_json
from extension_pvsg.m2.download import sha256
from extension_pvsg.m3.common import ROOT, runtime_versions
from extension_pvsg.validate_m0 import canonical_sha256


def check_runtime():
    if runtime_versions() != read_json(ROOT / "runtime.json"):
        raise ValueError("M3 runtime mismatch")


def check_features(cfg):
    root = Path(cfg["feature_cache"])
    if not (root / "summary.json").is_file():
        raise ValueError("M2 extraction is incomplete: run bash extension_pvsg/m2/launch_extract.sh first")
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
    if actual["config_sha256"] != canonical_sha256(feature_cfg):
        raise ValueError("M2 feature extraction config changed")
    from common.provenance import verify_artifact

    for name, expected_hash in actual["code_sha256"].items():
        verify_artifact(ROOT.parent / "m2" / name, expected_hash)
    for vid in sorted(expected):
        meta = read_json(root / f"{vid}.json")
        if (
            sha256(root / f"{vid}.npz") != summary["feature_hashes"][vid]
            or meta["feature_sha256"] != summary["feature_hashes"][vid]
            or meta["identity"] != actual
        ):
            raise ValueError(f"Feature shard corrupt/incompatible: {vid}")
    return sha256(root / "summary.json")
