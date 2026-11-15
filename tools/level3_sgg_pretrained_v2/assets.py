"""Pinned downloads and read-only verification. No model execution."""

import subprocess

from tools.level3_image_graph_v1.core import ROOT, read, sha256

CACHE = ROOT / "data/cache/level3-sgg-pretrained-v2"
LOCK = ROOT / "configs/level3/sgg-pretrained-assets-v2.json"


def verify(cache=CACHE):
    lock = read(LOCK)
    for name, spec in lock["repos"].items():
        path = cache / name
        head = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=all"], text=True
        )
        if head != spec["commit"] or dirty:
            raise ValueError("upstream_source_changed:" + name)
    for name, spec in lock["models"].items():
        for f in spec["files"]:
            path = cache / name / f["path"]
            if path.stat().st_size != f["size"] or sha256(path) != f["sha256"]:
                raise ValueError("checkpoint_changed:" + str(path))
    return lock


def prepare(cache=CACHE):
    import requests

    cache.mkdir(parents=True, exist_ok=True)
    lock = read(LOCK)
    for name, spec in lock["repos"].items():
        path = cache / name
        if not path.exists():
            subprocess.run(["git", "clone", spec["url"], str(path)], check=True)
            subprocess.run(["git", "-C", str(path), "checkout", "--detach", spec["commit"]], check=True)
    for name, spec in lock["models"].items():
        for f in spec["files"]:
            path = cache / name / f["path"]
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            with requests.get(f["url"], stream=True, timeout=120) as response:
                response.raise_for_status()
                with path.open("xb") as stream:
                    for block in response.iter_content(1024 * 1024):
                        stream.write(block)
    return verify(cache)
