"""CPU preflight and frozen, stratified pilot selection from existing train records."""

import hashlib
import importlib.metadata
import re
import subprocess
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

from PIL import Image

from src.knowledge.transe.core import sha256
from tools.magicbrush_joint_v1.schema import SCHEMA
from tools.magicbrush_joint_v1.storage import read, write_once
from tools.prepare_magicbrush import ROOT, digest

PACKAGE = Path(__file__).parent
DEFAULT_CONFIG = ROOT / "configs/level2/magicbrush-joint-qwen36-nvfp4-v1.json"
# Proxies for selection only. They NEVER create graph observations or operations.
BUCKETS = (
    ("removal", r"\b(remove|delete|erase|without|disappear)\b"),
    ("addition", r"\b(add|put|insert|place|include)\b"),
    ("replacement", r"\b(replace|instead|turn .* into|change .* to a)\b"),
    ("attribute", r"\b(color|colour|red|blue|green|yellow|black|white|larger|smaller)\b"),
    ("relationship", r"\b(under|above|behind|beside|holding|riding|on top|next to|left of|right of)\b"),
    ("difficult_or_other", r".*"),
)


def config(path=DEFAULT_CONFIG):
    result = read(path)
    if result["schema_version"] != "magicbrush_joint_config_v1" or result["batch_size"] != 1:
        raise ValueError("Only version 1, sequential batch 1 is supported")
    if type(result["seed"]) is not int or result["pilot_pairs"] != 20:
        raise ValueError("Pilot must freeze 20 pairs with an integer seed")
    if result["model"]["trust_remote_code"] or not re.fullmatch(r"[0-9a-f]{40}", result["model"]["revision"]):
        raise ValueError("Require immutable model revision and no remote-code execution")
    url = urlsplit(result["server"]["base_url"])
    if url.scheme != "http" or url.hostname not in ("localhost", "127.0.0.1", "::1") or url.path != "/v1":
        raise ValueError("Only an explicit local HTTP /v1 endpoint is allowed")
    if url.username or url.password or url.query or url.fragment:
        raise ValueError("Invalid local endpoint")
    if result["server"]["host"] != "127.0.0.1" or url.port != result["server"]["port"]:
        raise ValueError("Server binding differs from local endpoint")
    for key in ("max_attempts", "timeout_seconds", "max_response_bytes"):
        if type(result["http"][key]) is not int or result["http"][key] <= 0:
            raise ValueError("HTTP limits must be positive integers")
    if result["http"]["max_attempts"] > 5:
        raise ValueError("Retry budget exceeds five attempts")
    for key, maximum in (("max_nodes", 20), ("max_edges", 60), ("max_operations", 12)):
        if type(result["limits"][key]) is not int or not 1 <= result["limits"][key] <= maximum:
            raise ValueError("Graph budget exceeds downstream limits")
    if not 0.8 <= result["minimum_review_fraction"] <= 1:
        raise ValueError("Do not relax the established human quality gate")
    for key, base in (("output", ROOT / "data/processed/level2"), ("reviews", ROOT / "reviews")):
        p = (ROOT / result[key]).resolve()
        if not p.is_relative_to(base) or "joint" not in p.name or not p.name.endswith("-v1"):
            raise ValueError(f"{key} must use a new joint v1 output directory")
    return result


def validate_splits(manifest):
    index = {r["id"]: r for r in manifest["records"]}
    if len(index) != len(manifest["records"]):
        raise ValueError("duplicate_sample_id")
    membership, groups, pixels, byte_hashes = {}, {}, {}, {}
    if set(manifest["splits"]) != {"train", "val", "test"}:
        raise ValueError("unexpected_splits")
    for split, identifiers in manifest["splits"].items():
        for identifier in identifiers:
            if identifier in membership or identifier not in index:
                raise ValueError("duplicate_or_unknown_split_id")
            membership[identifier] = split
    if membership.keys() != index.keys():
        raise ValueError("incomplete_split_membership")
    for identifier, row in index.items():
        split = membership[identifier]
        if row["split"] != split or row["official_split"] != "train":
            raise ValueError("unexpected_dataset_split")
        for key in (row["source_group_id"], "session:" + row["img_id"]):
            if not key or groups.setdefault(key, split) != split:
                raise ValueError("source_group_or_session_split_leakage")
        for role in ("source_image", "target_image"):
            for table, key in ((pixels, "pixel_sha256"), (byte_hashes, "sha256")):
                checksum = row[role][key]
                if not checksum or table.setdefault(checksum, split) != split:
                    raise ValueError("image_split_leakage")
    return index


def bucket(instruction):
    return next(name for name, pattern in BUCKETS if re.search(pattern, instruction, re.I))


def select_pilot(manifest, seed):
    validate_splits(manifest)
    groups = {name: [] for name, _ in BUCKETS}
    for row in manifest["records"]:
        if row["split"] == "train":
            groups[bucket(row["instruction"])].append(row["id"])
    for rows in groups.values():
        rows.sort(key=lambda identifier: (digest(["joint-pilot-v1", seed, identifier]), identifier))
    selected = []
    # Round-robin keeps rare proxy strata present; no easy-case or graph filtering.
    while len(selected) < 20 and any(groups.values()):
        for rows in groups.values():
            if rows and len(selected) < 20:
                selected.append(rows.pop(0))
    if len(selected) != 20:
        raise ValueError("need_twenty_train_pairs")
    return selected


def image_check(row):
    for role in ("source_image", "target_image"):
        meta = row[role]
        path = ROOT / meta["path"]
        if sha256(path) != meta["sha256"]:
            raise ValueError("image_hash_mismatch:" + role)
        with Image.open(path) as image:
            image.load()
            rgb = image.convert("RGB")
            pixels = hashlib.sha256(f"{image.width}:{image.height}:RGB:".encode() + rgb.tobytes()).hexdigest()
            if pixels != meta["pixel_sha256"]:
                raise ValueError("image_pixel_hash_mismatch:" + role)
            if (image.width, image.height) != (meta["width"], meta["height"]):
                raise ValueError("image_dimensions_mismatch:" + role)


def source_hashes():
    files = list(PACKAGE.glob("*.py")) + [PACKAGE / "prompt_v1.txt"]
    files += [
        ROOT / name
        for name in (
            "tools/prepare_magicbrush.py",
            "tools/review_magicbrush.py",
            "tools/edit_model_v1/codec.py",
            "tools/edit_model_v1/data.py",
            "src/editing/operations.py",
            "src/editing/evaluation.py",
            "src/editing/splits.py",
            "src/knowledge/transe/core.py",
            "scripts/prepare_knowledge.py",
        )
    ]
    files += list((ROOT / "tools/magicbrush_review_assets").glob("*"))
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(files)}


def revision():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def identity(cfg):
    manifest_path = ROOT / cfg["manifest"]
    manifest = read(manifest_path)
    ids = select_pilot(manifest, cfg["seed"])
    return {
        "schema_version": "magicbrush_joint_run_v1",
        "config": cfg,
        "dataset_manifest_sha256": sha256(manifest_path),
        "dataset_revision": manifest["revision"],
        "pilot_ids": ids,
        "selection": "20 train pairs; round-robin intent-regex strata then SHA256(seed,id); proxies, not observed edit labels; no quality filtering",
        "selection_strata": dict(Counter(bucket(r["instruction"]) for r in manifest["records"] if r["id"] in ids)),
        "source_sha256": source_hashes(),
        "schema_sha256": digest(SCHEMA),
        "prompt_version": "joint-edit-focused-v1",
        "code_revision": revision(),
        "graph_scope": "edit-focused partial object/attribute/relation graphs; not exhaustive scene graphs",
        "automatic_gpu_launch": False,
        "automatic_queue_continuation": False,
    }


def prepare(cfg):
    run = identity(cfg)
    out = ROOT / cfg["output"]
    with_lock = out / "run.json"
    # Existing runs must retain their initial commit even after unrelated commits.
    if with_lock.exists():
        previous = read(with_lock)
        run["code_revision"] = previous["code_revision"]
    write_once(with_lock, run)
    snapshot = read(ROOT / cfg["manifest"])
    snapshot.update(review_ids=run["pilot_ids"], selection=run["selection"])
    write_once(out / "manifest.json", snapshot)
    return run


def check_run(cfg):
    out = ROOT / cfg["output"]
    run = read(out / "run.json")
    current = identity(cfg)
    current["code_revision"] = run["code_revision"]
    if run != current:
        raise ValueError("run_provenance_changed; use a new config/output version")
    snapshot = read(ROOT / cfg["manifest"])
    snapshot.update(review_ids=run["pilot_ids"], selection=run["selection"])
    if read(out / "manifest.json") != snapshot:
        raise ValueError("run_dataset_snapshot_changed")
    return run, snapshot


def plan(cfg):
    run = identity(cfg)
    manifest = read(ROOT / cfg["manifest"])
    index = validate_splits(manifest)
    errors = {}
    for identifier in run["pilot_ids"]:
        try:
            image_check(index[identifier])
        except (OSError, ValueError) as error:
            errors[identifier] = str(error)
    try:
        installed = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        installed = None
    return {
        **run,
        "stage": "CPU_only_preflight_no_model_download",
        "pilot_image_errors": errors,
        "local_vllm_version": installed,
        "runtime_verified": False,
        "two_image_gpu_smoke_verified": False,
        "human_reviewed_accuracy": None,
        "pairs_available": {k: len(v) for k, v in manifest["splits"].items()},
        "ready_for_cpu_prepare": not errors,
    }
