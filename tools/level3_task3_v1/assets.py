"""Pinned pretrained weights only; preparation never invokes inference."""

from tools.level3_image_graph_v1.core import read, sha256

from .core import ASSETS, CACHE


def verify():
    lock = read(ASSETS)
    for f in lock["files"]:
        p = CACHE / f["path"]
        if not p.is_file() or p.stat().st_size != f["size"] or sha256(p) != f["sha256"]:
            raise ValueError("asset_missing_or_changed:" + f["path"])
    return lock


def prepare():
    import requests

    for f in read(ASSETS)["files"]:
        p = CACHE / f["path"]
        if p.exists():
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(f["url"], stream=True, timeout=180) as r:
            r.raise_for_status()
            with p.open("xb") as out:
                for block in r.iter_content(1024 * 1024):
                    out.write(block)
    return verify()
