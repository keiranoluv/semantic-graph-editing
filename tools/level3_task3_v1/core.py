"""Immutable artifact and human-authored intent contracts, no GPU imports."""

import copy
import re
import subprocess
from pathlib import Path

from tools.level3_image_graph_v1.core import DEFAULT_LOCKED, ROOT, read, sha256, validate_record, write
from tools.level3_sgg_pretrained_v2.runner import verified_bundle

CONFIG = ROOT / "configs/level3/task3-pilot-v1.json"
DRAFTS = ROOT / "configs/level3/task3-drafts-v1.json"
ASSETS = ROOT / "configs/level3/task3-assets-v1.json"
CACHE = ROOT / "data/cache/level3-task3-v1"
RUNS = ROOT / "runs/level3/task3-v1"


def output_dir(path):
    path = Path(path).resolve()
    if RUNS.resolve() not in path.parents:
        raise ValueError("output_must_be_under_runs_level3_task3_v1")
    path.mkdir(parents=True, exist_ok=False)
    return path


def finish(path, mode):
    write(
        path / "receipt.json",
        {
            "schema": "level3-task3-receipt-v1",
            "mode": mode,
            "execution_complete": True,
            "files_sha256": {
                str(p.relative_to(path)): sha256(p)
                for p in sorted(path.rglob("*"))
                if p.is_file() and "wandb" not in p.relative_to(path).parts
            },
        },
    )


def sources():
    cfg = read(CONFIG)
    run = ROOT / cfg["source_run"]
    if sha256(run / "receipt.json") != cfg["source_receipt_sha256"]:
        raise ValueError("source_REACT_run_changed")
    if sha256(ROOT / "configs/level3/sgg-pretrained-v2.json") != cfg["parent_config_sha256"]:
        raise ValueError("source_protocol_changed")
    records = verified_bundle(run, "records.json")
    if [r["candidate_id"] for r in records] != cfg["image_ids"] or len(records) != 5:
        raise ValueError("exact_five_source_images_required")
    for r in records:
        validate_record(r)
        if r["status"] != "ok" or r["model"] != "react++":
            raise ValueError("source_must_be_real_REACT_prediction")
        if sha256(DEFAULT_LOCKED / r["image"]["image_path"]) != r["source_sha256"]:
            raise ValueError("source_image_changed")
    return cfg, {r["candidate_id"]: r for r in records}


def norm_box(record, node_id):
    b = record["grounding"]["nodes"][node_id]["bbox_xyxy"]
    im = record["image"]
    return [b[0] / im["width"], b[1] / im["height"], b[2] / im["width"], b[3] / im["height"]]


def region_ok(b):
    return (
        isinstance(b, list)
        and len(b) == 4
        and all(type(x) in (int, float) for x in b)
        and 0 <= b[0] < b[2] <= 1
        and 0 <= b[1] < b[3] <= 1
    )


def validate_tasks(pack, records, approved=False):
    if pack["schema"] != "level3-task3-tasks-v1" or not pack["tasks"]:
        raise ValueError("invalid_task_pack")
    ids = set()
    for task in pack["tasks"]:
        tid = task["task_id"]
        if not re.fullmatch(r"[a-z0-9-]+", tid) or tid in ids:
            raise ValueError("unsafe_or_duplicate_task_id")
        ids.add(tid)
        r = records[task["image_id"]]
        if task["source_sha256"] != r["source_sha256"]:
            raise ValueError("task_source_hash_mismatch")
        for k in ("instruction_en", "target_caption_en"):
            if not isinstance(task[k], str) or not task[k].strip() or len(task[k]) > 1500:
                raise ValueError("missing_or_oversized_text")
        if task["status"] not in ("draft", "approved"):
            raise ValueError("invalid_authoring_status")
        if approved and (task["status"] != "approved" or not task["reviewer"].strip()):
            raise ValueError("human_instruction_review_required")
        op = task["operation"]
        kind = op["kind"]
        nodes = {n["id"]: n for n in r["source_graph"]["nodes"]}
        fields = {
            "remove": {"kind", "source_node"},
            "replace": {"kind", "source_node", "label"},
            "add": {"kind", "label", "region"},
            "attribute": {"kind", "source_node", "attribute"},
            "relation": {"kind", "subject", "object", "predicate", "present"},
        }
        if kind not in fields or set(op) != fields[kind]:
            raise ValueError("invalid_operation_fields")
        if kind in ("remove", "replace", "attribute") and op["source_node"] not in nodes:
            raise ValueError("missing_source_anchor")
        if kind in ("add", "replace"):
            vocab = set(
                read(ROOT / "runs/level3/sgg-pretrained-v2/react-five-v1/model-metadata.json")["objects"].values()
            )
            if op["label"] not in vocab:
                raise ValueError("label_not_in_REACT_vocabulary")
            if kind == "replace" and op["label"] == nodes[op["source_node"]]["label"]:
                raise ValueError("replacement_must_change_category")
        if kind == "add" and not region_ok(op["region"]):
            raise ValueError("invalid_normalized_edit_region")
        if kind == "attribute" and not str(op["attribute"]).strip():
            raise ValueError("empty_attribute")
        if kind == "relation":
            if (
                op["subject"] not in nodes
                or op["object"] not in nodes
                or op["subject"] == op["object"]
                or type(op["present"]) is not bool
            ):
                raise ValueError("invalid_relation_operation")
            rels = set(
                read(ROOT / "runs/level3/sgg-pretrained-v2/react-five-v1/model-metadata.json")["relations"].values()
            ) - {"__background__"}
            if op["predicate"] not in rels:
                raise ValueError("unknown_predicate")
            exists = any(
                (e["subject"], e["predicate"], e["object"]) == (op["subject"], op["predicate"], op["object"])
                for e in r["source_graph"]["edges"]
            )
            if exists == op["present"]:
                raise ValueError("requested_relation_already_in_source")
    return pack


def code_identity():
    files = (
        list(Path(__file__).parent.glob("*.py")) + list(Path(__file__).parent.glob("*.html")) + [CONFIG, DRAFTS, ASSETS]
    )
    files += list((ROOT / "tools/level3_sgg_pretrained_v2").glob("*.py"))
    files += [
        ROOT / "requirements-level3-task3-v1.txt",
        ROOT / "src/editing/operations.py",
        ROOT / "tools/level3_image_graph_v1/core.py",
    ]
    return {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "files_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sorted(files)},
    }


def approved_pack(directory, records):
    pack = verified_bundle(Path(directory), "tasks.json")
    validate_tasks(pack, records, approved=True)
    if set(t["image_id"] for t in pack["tasks"]) != set(records):
        raise ValueError("approved_pilot_must_cover_all_five_images")
    return copy.deepcopy(pack)
