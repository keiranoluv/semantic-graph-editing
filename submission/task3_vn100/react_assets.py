"""CPU input integrity, white-alpha normalization and frozen REACT assets."""

import subprocess
from pathlib import Path

from common.paths import ROOT
from PIL import Image, ImageOps
from task3_vn100.image_graph import read, sha256


def verify_react():
    lock = read(LOCK)
    spec = lock["repos"]["SGG-Benchmark"]
    repo = CACHE / "SGG-Benchmark"
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True)
    if head != spec["commit"] or dirty:
        raise ValueError("official_REACT_code_changed")
    model = lock["models"]["react"]
    for f in model["files"]:
        p = CACHE / "react" / f["path"]
        if p.stat().st_size != f["size"] or sha256(p) != f["sha256"]:
            raise ValueError("REACT_asset_changed")
    return {"repo": spec, "model": model}


def normalize(path, output):
    """Match the reviewer's white RGBA composite; keep each image's pixel dimensions."""
    with Image.open(path) as im:
        original = {"mode": im.mode, "size": list(im.size), "sha256": sha256(path)}
        im = ImageOps.exif_transpose(im).convert("RGBA")
        rgb = Image.alpha_composite(Image.new("RGBA", im.size, "white"), im).convert("RGB")
        with Path(output).open("xb") as stream:
            rgb.save(stream, format="PNG")
    return {
        "native": original,
        "image_path": str(Path(output).resolve()),
        "sha256": sha256(output),
        "width": rgb.width,
        "height": rgb.height,
        "policy": "EXIF_transpose_RGBA_composite_white_RGB_original_size",
    }


CACHE = ROOT / "data/cache/level3-sgg-pretrained-v2"

LOCK = ROOT / "configs/level3/sgg-pretrained-assets-v2.json"
