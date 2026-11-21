"""Read-only CPU checks before the user launches feature extraction."""

import argparse
import json
import subprocess
from importlib.metadata import version
from pathlib import Path

from ..contracts import ROOT, read_json
from ..validate_m0 import validate_manifest
from .download import sha256

M2 = Path(__file__).resolve().parent
PROJECT = M2.parents[2]


def verify_locks():
    for filename in ("m0_status.json", "m1_status.json"):
        status = read_json(ROOT / "provenance" / filename)
        for name, expected in status["artifact_sha256"].items():
            if sha256(ROOT / name) != expected:
                raise ValueError(f"Frozen {filename} artifact changed: {name}")
    status = read_json(M2 / "provenance/m2_status.json")
    for name, expected in status["artifact_sha256"].items():
        if sha256(M2 / name) != expected:
            raise ValueError(f"M2 artifact changed: {name}")
    for name, expected in status["dataset_artifact_sha256"].items():
        if sha256(PROJECT / name) != expected:
            raise ValueError(f"Dataset artifact changed: {name}")
    for name, expected in status["runtime_versions"].items():
        if version(name) != expected:
            raise ValueError(f"Runtime version mismatch: {name} needs {expected}, found {version(name)}")
    config = read_json(M2 / "configs/clip_masked_v1.json")
    if sha256(PROJECT / "data/external/pvsg_clip_v1/ViT-B-32.pt") != config["checkpoint_sha256"]:
        raise ValueError("Checkpoint identity mismatch")
    for item in read_json(M2 / "manifests/media.lock.json")["archives"]:
        if (
            item["kind"] == "masks"
            and sha256(PROJECT / "data/external/pvsg_media_v1/archives" / item["filename"]) != item["sha256"]
        ):
            raise ValueError("Mask archive identity mismatch")
    manifest = read_json(M2 / "manifests/dataset.lock.json")
    validate_manifest(manifest, require_selected=True)
    audit = read_json(PROJECT / "data/external/pvsg_m2_cache_v4/media_audit.json")
    if audit["passed"] != 400 or audit["errors"]:
        raise ValueError("Dataset media audit is incomplete")
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cpu-review", action="store_true", help="Only review CPU readiness before commit; never launches GPU"
    )
    args = parser.parse_args()
    verify_locks()
    if not args.cpu_review:
        state = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all", "--", str(M2)], cwd=PROJECT, text=True
        )
        if state.strip():
            raise ValueError("M2 changes must be committed before a GPU run")
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(M2 / "features.py")],
            cwd=PROJECT,
            check=True,
            stdout=subprocess.DEVNULL,
        )
    print(
        json.dumps(
            {
                "preflight": "passed",
                "device_checked": "cpu",
                "GPU_started": False,
                "model": "CLIP ViT-B/32 frozen",
                "batch_size": 64,
                "seed": 17,
                "videos": 400,
                "output": "data/external/pvsg_clip_cache_v1",
                "resume": "verified completed videos only",
                "training": False,
                "automatic_next_model_or_training": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
