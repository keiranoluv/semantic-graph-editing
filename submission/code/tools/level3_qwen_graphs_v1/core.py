"""CPU input integrity, white-alpha normalization and frozen REACT assets."""

import subprocess
from pathlib import Path

from PIL import Image, ImageOps

from tools.level3_image_graph_v1.core import ROOT, read, sha256
from tools.level3_qwen21_v1.core import bundle
from tools.level3_sgg_pretrained_v2.assets import CACHE, LOCK

CONFIG = ROOT / "configs/level3/qwen-react100-v1.json"
RUNS = ROOT / "runs/level3/qwen-react100-v1"


def validate_rows(rows, run, expected=100):
    if len(rows) != expected or len({r["task_id"] for r in rows}) != expected:
        raise ValueError("expected_unique_source_target_pairs")
    if len({r["image_id"] for r in rows}) != expected:
        raise ValueError("duplicate_source_image")
    run = Path(run).resolve()
    for r in rows:
        if r["status"] != "ok" or "/" in r["task_id"] or ".." in r["task_id"]:
            raise ValueError("invalid_pair")
        for role in ("source", "target"):
            p = Path(r[role + "_path"]).resolve()
            checksum = r["source_sha256"] if role == "source" else r["target"]["sha256"]
            if run not in p.parents or sha256(p) != checksum:
                raise ValueError("pair_image_changed_or_outside_run")
    return rows


def inputs():
    cfg = read(CONFIG)
    parent = ROOT / cfg["input_run"]
    if sha256(parent / "receipt.json") != cfg["input_receipt_sha256"]:
        raise ValueError("Qwen_run_receipt_changed")
    if sha256(LOCK) != cfg["asset_lock_sha256"]:
        raise ValueError("asset_lock_changed")
    old_cfg = ROOT / "configs/level3/sgg-pretrained-v2.json"
    if sha256(old_cfg) != cfg["parent_sgg_config_sha256"] or read(old_cfg)["react"] != cfg["react"]:
        raise ValueError("REACT_thresholds_changed")
    if (
        cfg["expected_pairs"] != 100
        or cfg["batch_size"] != 1
        or cfg["seed"] != 42
        or cfg["training"]
        or cfg["automatic_next_run"]
    ):
        raise ValueError("fixed_protocol_changed")
    rows = bundle(parent, "results.json")
    if read(parent / "run.json")["config"]["generation"]["model"] != "Qwen/Qwen-Image-2.1":
        raise ValueError("Qwen_source_run_required")
    return cfg, validate_rows(rows, parent)


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


def new_output(path):
    p = Path(path).resolve()
    if RUNS.resolve() not in p.parents:
        raise ValueError("output_must_be_under_qwen_react100_v1")
    p.mkdir(parents=True, exist_ok=False)
    return p


def code_identity():
    files = list(Path(__file__).parent.glob("*.py")) + list(Path(__file__).parent.glob("*.html"))
    files += [
        CONFIG,
        LOCK,
        ROOT / "configs/level3/sgg-pretrained-v2.json",
        ROOT / "requirements-level3-sgg-pretrained-v2.txt",
        ROOT / "scripts/run_level3_qwen_graphs_v1.sh",
        ROOT / "tools/level3_image_graph_v1/core.py",
        ROOT / "src/editing/operations.py",
        ROOT / "tools/level3_sgg_pretrained_v2/backends.py",
        ROOT / "tools/level3_sgg_pretrained_v2/adapter.py",
        ROOT / "tools/level3_qwen21_v1/runner.py",
        ROOT / "tools/level3_qwen21_v1/core.py",
        ROOT / "tools/level3_qwen100_v1/receipt.py",
        ROOT / "tools/level3_qwen100_v1/viewer.py",
    ]
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--", *map(str, files)], cwd=ROOT, text=True)
    if dirty:
        raise ValueError("commit_related_code_before_inference")
    return {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "files_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in files},
    }
