"""Separate M4 config/provenance; frozen M0--M3 remain read-only dependencies."""

from pathlib import Path

from extension_pvsg.m2.download import sha256
from extension_pvsg.m3.common import identity as m3_identity

ROOT = Path(__file__).resolve().parent


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
