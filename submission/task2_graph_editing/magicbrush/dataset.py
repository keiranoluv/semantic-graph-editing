"""Read one pinned split at a time; rows contain both policies on the same input."""

import json
from pathlib import Path

from common.io import read, sha256

HERE = Path(__file__).resolve().parent

DATA = HERE / "data"

MANIFEST = DATA / "manifest.json"


def rows(split, *, held=False):
    if split not in {"train", "val", "test"}:
        raise ValueError(split)
    manifest = read(MANIFEST)
    result = []
    for name, expected in sorted(manifest["files"].items()):
        if not name.startswith(split + "-") or name.endswith("-held.jsonl") != held:
            continue
        path = DATA / name
        if sha256(path) != expected:
            raise ValueError(f"Dataset checksum changed: {name}")
        result.extend(json.loads(line) for line in path.read_text().splitlines())
    ids = {row["id"] for row in result if row["split"] == split}
    count = manifest["counts"][split]["held" if held else "kept"]
    if len(result) != count or len(ids) != count:
        raise ValueError(f"Incomplete or duplicated {split} partition")
    return result
