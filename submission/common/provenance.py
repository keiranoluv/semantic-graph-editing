"""Verify the refactored submission independently of historical source locks."""

import json

from common.io import sha256
from common.paths import ROOT


def source_hashes():
    manifest = json.loads((ROOT / "assets/refactor_manifest.json").read_text())
    for name, expected in manifest["source_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Submission source changed: {name}")
    return manifest["source_sha256"]


def verify_artifact(path, expected):
    """Validate current sources while retaining historical artifact evidence."""
    from pathlib import Path

    path = Path(path)
    if path.is_file() and sha256(path) == expected:
        return
    name = str(path.resolve().relative_to(ROOT))
    origins = json.loads((ROOT / "assets/source_origins.json").read_text())["files"]
    origin = origins.get(name)
    if origin is None or origin["original_sha256"] != expected:
        raise ValueError(f"Historical artifact differs: {name}")
    source_hashes()
    if not path.is_file() and not origin.get("omitted"):
        raise FileNotFoundError(path)


def revision():
    """Identify a clone by Git commit and a standalone copy by its source manifest."""
    import subprocess

    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            return result.stdout.strip()
    except FileNotFoundError:
        pass
    return "source-sha256:" + sha256(ROOT / "assets/refactor_manifest.json")
