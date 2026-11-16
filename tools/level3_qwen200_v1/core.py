"""Versioned extension contracts; old manifests, images and outcomes stay immutable."""

import re
import subprocess
from collections import Counter
from pathlib import Path

from tools.level3_image_graph_v1.core import ROOT, read, sha256
from tools.level3_qwen21_v1.core import LOCK, assets, bundle
from tools.level3_qwen21_v1.runner import runtime
from tools.level3_qwen100_v1.core import identity as parent_identity
from tools.level3_qwen100_v1.core import reuse
from tools.level3_qwen100_v1.runner import preflight as parent_preflight

CONFIG = ROOT / "configs/level3/qwen200-extension-v1.json"
MANIFEST = ROOT / "configs/level3/qwen200-extension-instructions-v1.json"
COHORT = ROOT / "configs/level3/qwen200-cohort-v1.json"
RUNS = ROOT / "runs/level3/qwen200-extension-v1"
COUNTS = {"add": 40, "remove": 38, "replace": 12, "attribute": 10}
SOURCE_FIELDS = (
    "image_id",
    "dataset",
    "sha256",
    "dhash",
    "source_path",
    "relative_path",
    "source_revision_sha",
    "source_asset_url",
    "width",
    "height",
    "dataset_license_claim",
    "per_image_rights_verified",
    "local_evaluation_only",
)


def normalized(text):
    return " ".join(text.casefold().split())


def validate_extension(pack, parent, cohort):
    rows, old = pack["tasks"], parent["tasks"]
    if len(rows) != 100 or len(old) != 100 or Counter(r["dataset"] for r in rows) != {"sidewalk": 50, "travel": 50}:
        raise ValueError("exact_100_extension_50_50_required")
    if len(cohort["tasks"]) != 200 or cohort["tasks"][:100] != old or cohort["tasks"][100:] != rows:
        raise ValueError("combined_manifest_must_preserve_old_100_exactly")
    if Counter(t["image_id"] for t in cohort["tasks"]) != {r["image_id"]: 2 for r in old}:
        raise ValueError("two_independent_requests_per_source_required")
    for field in ("task_id", "image_id", "sha256"):
        if len({r[field] for r in rows}) != 100:
            raise ValueError("duplicate_extension_" + field)
    if set(r["task_id"] for r in rows) & set(r["task_id"] for r in old):
        raise ValueError("extension_must_not_reuse_old_task_ids")
    old_prompts = {normalized(r["instruction_en"]) for r in old}
    prompts = [normalized(r["instruction_en"]) for r in rows]
    if len(set(prompts)) != 100 or set(prompts) & old_prompts:
        raise ValueError("duplicate_new_instruction_text")
    if Counter(r["operation"]["kind"] for r in rows) != COUNTS:
        raise ValueError("fixed_operation_mix_changed")
    if Counter(r["dataset"] for r in rows if r["operation"]["kind"] == "attribute") != {"sidewalk": 5, "travel": 5}:
        raise ValueError("five_color_probes_per_dataset_required")
    if pack["source_revisions"] != parent["source_revisions"]:
        raise ValueError("source_revision_changed")
    for i, (row, prior) in enumerate(zip(rows, old, strict=True), 1):
        if any(row[k] != prior[k] for k in SOURCE_FIELDS):
            raise ValueError("extension_must_use_original_source_not_target")
        if (
            row["parent_task_id"] != prior["task_id"]
            or row["source_index"] != i
            or row["request_index"] != i + 100
            or row["task_id"] != prior["image_id"] + "-q200-2"
            or not re.fullmatch(r"[a-zA-Z0-9_-]+", row["task_id"])
        ):
            raise ValueError("invalid_extension_pair_identity")
        if row["author"] != "assistant_visual_review" or row["human_outcome_label"] is not None:
            raise ValueError("do_not_forge_human_outcome")
        if not row["visual_evidence"].strip() or not row["instruction_en"].strip():
            raise ValueError("visual_authoring_required")
        if row["input_policy"] != "independent_edit_of_original_source_not_previous_target":
            raise ValueError("sequential_edits_not_authorized")
        op = row["operation"]
        required = {"kind", "label", "location"}
        if op["kind"] == "replace":
            required.add("replacement_label")
            if normalized(op["label"]) == normalized(op["replacement_label"]):
                raise ValueError("replacement_must_change_category")
        elif op["kind"] == "attribute":
            required |= {"attribute", "source_value", "target_value"}
            if op["attribute"] != "color" or normalized(op["source_value"]) == normalized(op["target_value"]):
                raise ValueError("color_change_required")
        if set(op) != required or any(not isinstance(v, str) or not v.strip() for v in op.values()):
            raise ValueError("invalid_operation_contract")
        is_color = op["kind"] == "attribute"
        if row["evaluation_group"] != ("color_attribute_probe" if is_color else "object_edit") or row[
            "sgg_edit_fulfillment_support"
        ] != ("unsupported_color_attribute" if is_color else "requires_instance_grounding_and_detection"):
            raise ValueError("color_probe_support_must_be_explicit")
    return rows


def preflight(check_assets=True):
    cfg = read(CONFIG)
    for key in ("manifest", "cohort_manifest", "parent_config", "parent_manifest"):
        if sha256(ROOT / cfg[key]) != cfg[key + "_sha256"]:
            raise ValueError("frozen_" + key + "_changed")
    if ROOT / cfg["manifest"] != MANIFEST or ROOT / cfg["cohort_manifest"] != COHORT:
        raise ValueError("unexpected_manifest_path")
    parent_cfg, old = parent_preflight(False)  # verifies original source bytes and source receipts
    parent = read(ROOT / cfg["parent_manifest"])
    pack, cohort = read(MANIFEST), read(COHORT)
    if (
        pack["parent_manifest_sha256"] != cfg["parent_manifest_sha256"]
        or any(cohort[k] != cfg[k] for k in ("parent_manifest", "parent_manifest_sha256"))
        or cohort["extension_manifest_sha256"] != sha256(MANIFEST)
    ):
        raise ValueError("manifest_parent_link_changed")
    if (
        cfg["generation"] != parent_cfg["generation"]
        or cfg["batch_size"] != 1
        or cfg["expected_count"] != 100
        or cfg["cohort_request_count"] != 200
        or cfg["unique_source_count"] != 100
        or cfg["expected_operation_counts"] != COUNTS
        or cfg["color_probe_count"] != 10
        or cfg["training"]
        or cfg["automatic_next_run"]
        or cfg["quality_retries"]
        or cfg["training_or_graph_scoring"]
        or cfg["wandb_mode"] != "online"
        or cfg["seed"] != 42
        or sha256(LOCK) != cfg["asset_lock_sha256"]
    ):
        raise ValueError("fixed_extension_protocol_changed")
    rows = validate_extension(pack, parent, cohort)
    previous_run = ROOT / cfg["parent_completed_run"]
    if sha256(previous_run / "receipt.json") != cfg["parent_completed_run_receipt_sha256"]:
        raise ValueError("old_run_receipt_changed")
    done = bundle(previous_run, "results.json")
    if len(done) != 100 or any(
        r["task_id"] != t["task_id"] or r["status"] != "ok" or r["prompt"] != t["instruction_en"]
        for r, t in zip(done, old, strict=True)
    ):
        raise ValueError("original_100_successes_required")
    if check_assets:
        assets()
        runtime()
    return cfg, rows


def identity():
    files = set(parent_identity()["files_sha256"])
    files |= {str(p.relative_to(ROOT)) for p in Path(__file__).parent.glob("*.py")}
    files |= {
        str(p.relative_to(ROOT))
        for p in (
            CONFIG,
            MANIFEST,
            COHORT,
            ROOT / "scripts/run_level3_qwen200_extension_v1.sh",
            ROOT / "tools/level3_qwen_review_v2.html",
        )
    }
    if subprocess.check_output(["git", "status", "--porcelain", "--", *sorted(files)], cwd=ROOT, text=True):
        raise ValueError("commit_extension_code_before_GPU_inference")
    return {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "files_sha256": {f: sha256(ROOT / f) for f in sorted(files)},
    }


def new_output(path):
    path = Path(path).resolve()
    if RUNS.resolve() not in path.parents:
        raise ValueError("output_outside_extension_run_root")
    path.mkdir(parents=True, exist_ok=False)
    return path


def resume_cases(path, ident, tasks):
    path = Path(path).resolve()
    if RUNS.resolve() not in path.parents:
        raise ValueError("resume_must_be_extension_run_not_old_100")
    previous = reuse(path, ident)
    index = {t["task_id"]: t for t in tasks}
    if set(previous) - set(index):
        raise ValueError("unknown_resume_case")
    for tid, row in previous.items():
        task = index[tid]
        if (
            row["image_id"] != task["image_id"]
            or row["prompt"] != task["instruction_en"]
            or row["native_source_sha256"] != task["sha256"]
            or row["completed_steps"] != 40
        ):
            raise ValueError("resume_case_intent_changed")
        if sha256(row["source_path"]) != row["source_sha256"] or sha256(row["target_path"]) != row["target"]["sha256"]:
            raise ValueError("resume_image_bytes_changed")
    return previous


def json_data(value):
    import json

    data = json.dumps(value, ensure_ascii=False)
    for c, escaped in [("<", "\\u003c"), (">", "\\u003e"), ("&", "\\u0026")]:
        data = data.replace(c, escaped)
    return data
