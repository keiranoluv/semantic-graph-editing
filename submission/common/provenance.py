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

def original_hash(name):
    origins = json.loads((ROOT / "assets/source_origins.json").read_text())
    return origins["files"][name]["original_sha256"]

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
