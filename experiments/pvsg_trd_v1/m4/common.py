"""Separate M4 config/provenance; frozen M0--M3 remain read-only dependencies."""

import subprocess
from pathlib import Path

from ..contracts import read_json
from ..m2.download import sha256
from ..m3.common import identity as m3_identity
from ..m3.common import require_committed as require_m3_committed
from ..validate_m0 import canonical_sha256

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
# Share the M3 GPU mutex. An M3 queue and an M4 job cannot train concurrently.
GPU_LOCK = PROJECT / "reports/pvsg-trd-m3-v1/.gpu.lock"


def config():
    cfg = read_json(ROOT / "configs/ablations_v1.json")
    parent = read_json(ROOT.parent / "m3/configs/baselines_v1.json")
    exceptions = {"experiment_id", "models", "output_root"}
    for key, value in parent.items():
        if key not in exceptions and cfg[key] != value:
            raise ValueError(f"M4 comparison must preserve M3 setting: {key}")
    if set(cfg["models"]) != set(cfg["variants"]):
        raise ValueError("Variant registry mismatch")
    return cfg


def code_identity():
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(ROOT.rglob("*.py")) if not p.name.startswith("test_")}


def identity(cfg, feature_sha):
    parent = m3_identity(cfg, feature_sha)
    return {
        **parent,
        "m3_code": parent["code"],
        "code": code_identity(),
        "m3_status_sha256": sha256(ROOT.parent / "m3/provenance/m3_status.json"),
        "sources_sha256": sha256(ROOT / "provenance/sources.json"),
    }


def verify_parent():
    root = ROOT.parent / "m3"
    status = read_json(root / "provenance/m3_status.json")
    for name, expected in status["artifact_sha256"].items():
        if sha256(root / name) != expected:
            raise ValueError(f"Frozen M3 artifact changed: {name}")


def verify_self():
    status = read_json(ROOT / "provenance/m4_status.json")
    for name, expected in status["artifact_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Frozen M4 artifact changed: {name}")


def require_committed():
    require_m3_committed()
    state = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", str(ROOT)], cwd=PROJECT, text=True
    )
    if state.strip():
        raise ValueError("Commit M4 code before any GPU job")
    subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(ROOT / "models.py")],
        cwd=PROJECT,
        check=True,
        stdout=subprocess.DEVNULL,
    )


def signature(cfg, feature_sha):
    return canonical_sha256(identity(cfg, feature_sha))
