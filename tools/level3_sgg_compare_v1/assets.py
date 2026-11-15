"""Pinned official assets, CPU verification, no remote code or image uploads."""

import importlib.metadata
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path

from tools.level3_image_graph_v1.core import ROOT, digest, read, sha256

LOCK = ROOT / "configs/level3/sgg-assets-v1.json"
CACHE = ROOT / "data/cache/level3-sgg-v1"


def verify_source(path, spec):
    path = Path(path).resolve()
    actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if actual != spec["commit"]:
        raise ValueError("upstream_commit_mismatch")
    files = subprocess.check_output(
        ["git", "-C", str(path), "ls-tree", "-r", "--name-only", "HEAD"], text=True
    ).splitlines()
    hashes = {}
    for name in files:
        file = path / name
        if file.is_symlink() or file.read_bytes() != subprocess.check_output(
            ["git", "-C", str(path), "show", f"HEAD:{name}"]
        ):
            raise ValueError("upstream_file_changed:" + name)
        hashes[name] = sha256(file)
    if any(p.relative_to(path).as_posix() not in hashes for p in path.rglob("*.py")):
        raise ValueError("unexpected_upstream_python_file")
    return {"commit": actual, "files_sha256": hashes}


def prepare_assets(cache=CACHE):
    cache = Path(cache)
    lock = read(LOCK)
    cache.mkdir(parents=True, exist_ok=True)
    repo = cache / "SyntheticVG"
    spec = lock["syntheticvg"]
    if not repo.exists():
        subprocess.run(["git", "clone", spec["url"], str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "checkout", "--detach", spec["commit"]], check=True)
    verify_source(repo, spec)
    for row in [
        *({**f, "destination": "robin/" + f["path"]} for f in lock["robin"]["files"]),
        {**lock["sam"], "destination": "sam_vit_h_4b8939.pth"},
    ]:
        path = cache / row["destination"]
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            part = path.with_suffix(path.suffix + ".download")
            print("Downloading", row["destination"], flush=True)
            with part.open("xb") as stream, urllib.request.urlopen(row["url"], timeout=120) as response:
                shutil.copyfileobj(response, stream)
            if part.stat().st_size != row["size"] or sha256(part) != row["sha256"]:
                raise ValueError("asset_download_hash_mismatch")
            part.rename(path)
        if sha256(path) != row["sha256"]:
            raise ValueError("asset_hash_mismatch:" + str(path))
    return verify_assets(cache)


def verify_assets(cache=CACHE):
    cache = Path(cache)
    lock = read(LOCK)
    source = verify_source(cache / "SyntheticVG", lock["syntheticvg"])
    files = {"robin/" + f["path"]: f["sha256"] for f in lock["robin"]["files"]}
    files["sam_vit_h_4b8939.pth"] = lock["sam"]["sha256"]
    for name, expected in files.items():
        if sha256(cache / name) != expected:
            raise ValueError("asset_hash_mismatch:" + name)
    return {
        "lock_sha256": sha256(LOCK),
        "source": source,
        "asset_sha256": files,
        "robin_revision": lock["robin"]["revision"],
        "training": False,
    }


def runtime_versions():
    expected = {
        "torch": "2.5.1",
        "torchvision": "0.20.1",
        "transformers": "4.43.3",
        "open_clip_torch": "2.24.0",
        "timm": "0.9.16",
        "accelerate": "0.33.0",
        "pycocotools": "2.0.10",
        "Pillow": "11.2.1",
        "segment-anything": "1.0",
    }
    actual = {k: importlib.metadata.version(k) for k in expected}
    if any(actual[k].split("+")[0] != v for k, v in expected.items()):
        raise ValueError("use_isolated_pinned_robin_environment:" + json.dumps(actual))
    dist = importlib.metadata.distribution("segment-anything")
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    if direct.get("vcs_info", {}).get("commit_id") != read(LOCK)["sam"]["code_commit"]:
        raise ValueError("sam_source_revision_mismatch")
    actual["sam_code_sha256"] = digest(
        {str(p): sha256(dist.locate_file(p)) for p in dist.files if str(p).endswith(".py")}
    )
    return actual
