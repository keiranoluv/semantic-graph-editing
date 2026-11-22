"""CPU-only contracts for paired inputs, frozen assets and bounded execution."""

from pathlib import Path

from common.paths import ROOT
from task3_vn100.image_graph import read, sha256, write

CONFIG = ROOT / "configs/level3/qwen21-pilot-v1.json"

LOCK = ROOT / "configs/level3/qwen21-assets-v1.json"

CACHE = ROOT / "data/cache/level3-qwen21-v1"


def bundle(path, name):
    path = Path(path).resolve()
    receipt = read(path / "receipt.json")
    if not receipt["execution_complete"] or name not in receipt["files_sha256"]:
        raise ValueError("incomplete_or_unbound_bundle")
    for f, checksum in receipt["files_sha256"].items():
        target = (path / f).resolve()
        if path not in target.parents or sha256(target) != checksum:
            raise ValueError("changed_artifact:" + f)
    return read(path / name)


def finish(path, success=True):
    write(
        path / "receipt.json",
        {
            "schema": "qwen21-receipt-v1",
            "execution_complete": success,
            "files_sha256": {str(p.relative_to(path)): sha256(p) for p in sorted(path.rglob("*")) if p.is_file()},
        },
    )


def assets():
    lock = read(LOCK)
    if lock["repo_id"] != "Qwen/Qwen-Image-2.1" or lock["revision"] != read(CONFIG)["revision"]:
        raise ValueError("checkpoint_revision_mismatch")
    root = (CACHE / "model").resolve()
    for f in lock["files"]:
        p = (root / f["path"]).resolve()
        if root not in p.parents or p.stat().st_size != f["size"] or sha256(p) != f["sha256"]:
            raise ValueError("asset_changed:" + f["path"])
    return lock
