"""Frozen AI-authored intent, image provenance and immutable result contracts."""
import subprocess
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps

from tools.level3_image_graph_v1.core import ROOT, digest, read, sha256
from tools.level3_qwen21_v1.core import LOCK, bundle

CONFIG = ROOT / "configs/level3/qwen100-v1.json"
MANIFEST = ROOT / "configs/level3/qwen100-instructions-v1.json"
RUNS = ROOT / "runs/level3/qwen100-v1"


def new_output(path):
    path = Path(path).resolve()
    if RUNS.resolve() not in path.parents:
        raise ValueError("output_outside_versioned_run_root")
    path.mkdir(parents=True, exist_ok=False)
    return path


def validate(pack, verify_images=True):
    rows = pack["tasks"]
    if len(rows) != 100 or Counter(r["dataset"] for r in rows) != {"sidewalk": 50, "travel": 50}:
        raise ValueError("exact_100_unique_sources_50_50_required")
    for field in ["task_id", "image_id", "sha256"]:
        if len({r[field] for r in rows}) != 100:
            raise ValueError("duplicate_" + field)
    for r in rows:
        if r["author"] != "assistant_visual_review" or r["human_outcome_label"] is not None:
            raise ValueError("do_not_forge_human_annotation")
        if not r["visual_evidence"].strip() or not r["instruction_en"].strip():
            raise ValueError("missing_visual_authoring")
        if r["operation"]["kind"] not in ("add", "remove", "replace"):
            raise ValueError("unsupported_operation")
        if not r["operation"]["label"] or not r["operation"]["location"]:
            raise ValueError("missing_grounded_intent")
        if r["source_revision_sha"] != pack["source_revisions"][r["dataset"]]:
            raise ValueError("unpinned_source")
        path = Path(r["source_path"]).resolve()
        if (ROOT / "data/raw/level3/qwen21-batch100-v1").resolve() not in path.parents:
            raise ValueError("source_outside_new_dataset_root")
        if verify_images:
            if sha256(path) != r["sha256"]:
                raise ValueError("source_bytes_changed")
            with Image.open(path) as im:
                im.verify()
        if "/" in r["task_id"] or ".." in r["task_id"]:
            raise ValueError("unsafe_task_id")
    for i, a in enumerate(rows):
        for b in rows[i + 1 :]:
            if (int(a["dhash"], 16) ^ int(b["dhash"], 16)).bit_count() <= 4:
                raise ValueError("near_duplicate_source_pair")
    return rows


def identity():
    files = list(Path(__file__).parent.glob("*.py"))
    files += list((ROOT / "tools/level3_qwen21_v1").glob("*.py"))
    files += [
        CONFIG,
        MANIFEST,
        LOCK,
        ROOT / "requirements-level3-qwen100-v1.txt",
        ROOT / "requirements-level3-qwen21-v1.txt",
        ROOT / "scripts/run_level3_qwen100_v1.sh",
        ROOT / "tools/level3_image_graph_v1/core.py",
    ]
    return {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "files_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sorted(files)},
    }


def reuse(path, run_identity):
    """Reuse successful case receipts from an interrupted run; never rewrite it."""
    path = Path(path)
    old = read(path / "run.json")
    for key in ["manifest_sha256", "config_sha256", "asset_lock_sha256"]:
        if old[key] != run_identity[key]:
            raise ValueError("resume_protocol_changed")
    if old["code"]["files_sha256"] != run_identity["code"]["files_sha256"]:
        raise ValueError("resume_code_changed")
    out = {}
    for p in sorted((path / "cases").glob("*/receipt.json")):
        r = bundle(p.parent, "result.json")
        if r["status"] == "ok":
            out[r["task_id"]] = r
    # Prior reused successes can be chained without rewriting original case artifacts.
    for r in read(path / "results.json") if (path / "results.json").exists() else []:
        if r["status"] == "ok" and r["task_id"] not in out:
            verified = bundle(Path(r["case_dir"]), "result.json")
            if digest(verified) != digest(r):
                raise ValueError("reused_result_changed")
            out[r["task_id"]] = r
    return out


def source_png(row, dest):
    with Image.open(row["source_path"]) as im:
        ImageOps.exif_transpose(im).convert("RGB").save(dest)
    return sha256(dest)
