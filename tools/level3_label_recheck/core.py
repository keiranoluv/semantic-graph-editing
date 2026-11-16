import copy
import json
import os
import subprocess
from pathlib import Path

from tools.level3_evaluation.core import ROOT, inputs
from tools.level3_evaluation.report import FIELDS, validate_reviews
from tools.level3_image_graph_v1.core import read, sha256
from tools.level3_qwen_unified.core import atomic

DEFAULT_OUTPUT = ROOT / "runs/level3/evaluation-v1/seed42/label-recheck-v1"


def normalize(pack, records, cohort_sha256, blank_is_pass=False):
    """Only explicit JSON null fields in present, image-bound rows may be defaulted."""
    validate_reviews(pack, records, cohort_sha256, allow_partial=True)
    if len(pack["reviews"]) != len(records):
        raise ValueError("all_200_rows_required_missing_rows_are_not_pass")
    rows = {r["task_id"]: r for r in pack["reviews"]}
    result = copy.deepcopy(pack)
    result["reviews"] = []
    defaults, failures = {}, []
    counts = {"fulfillment_only": 0, "preservation_only": 0, "both": 0}
    for record in records:
        tid = record["task"]["task_id"]
        row = copy.deepcopy(rows[tid])
        empty = [k for k in FIELDS if row[k] is None]
        if empty and not blank_is_pass:
            raise ValueError("blank_is_pass_requires_explicit_user_convention")
        if empty:
            defaults[tid] = empty
            for k in empty:
                row[k] = True
        row["status"] = "reviewed"
        result["reviews"].append(row)
        a, b = row["edit_fulfillment"], row["preservation"]
        if not a or not b:
            failures.append(tid)
            counts["both" if not a and not b else "fulfillment_only" if not a else "preservation_only"] += 1
    result["normalization"] = {
        "policy": "each_explicit_null_field_means_pass_by_user_instruction",
        "authorization": "User specified that either or both blank criteria mean pass; explicit true/false values stay unchanged.",
        "defaulted_fields": defaults,
        "missing_rows_defaulted": False,
        "phase": "awaiting_user_recheck_not_final_report",
    }
    validate_reviews(result, records, cohort_sha256)
    return (
        result,
        failures,
        {
            "cases": len(records),
            "failures": len(failures),
            "passes": len(records) - len(failures),
            "failure_types": counts,
            "rows_with_defaults": len(defaults),
            "fields_defaulted": sum(map(len, defaults.values())),
            "automatic_metric_scores_read": False,
        },
    )


def prepare(review_path, output, blank_is_pass=False):
    cfg, records = inputs()
    review_path = Path(review_path).resolve()
    raw = review_path.read_bytes()
    raw_sha = sha256(review_path)
    pack, failures, summary = normalize(json.loads(raw), records, sha256(ROOT / cfg["cohort"]), blank_is_pass)
    out = Path(output).resolve()
    if (ROOT / "runs/level3/evaluation-v1").resolve() not in out.parents:
        raise ValueError("output_outside_evaluation_runs")
    if out.exists() and any(out.iterdir()):
        raise ValueError("refuse_existing_recheck_output_use_serve_or_new_OUTPUT")
    files = list(Path(__file__).parent.glob("*.py")) + [
        Path(__file__).with_name("viewer.html"),
        ROOT / "scripts/review_level3_labels.sh",
    ]
    if subprocess.check_output(["git", "status", "--porcelain", "--", *map(str, files)], cwd=ROOT):
        raise ValueError("commit_recheck_code_before_preparing_real_labels")
    code = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "files_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in files},
    }
    pack["normalization"].update(raw_sha256=raw_sha, raw_filename=review_path.name)
    out.mkdir(parents=True)
    (out / "original-reviewed.json").write_bytes(raw)
    atomic(out / "normalized-reviewed.json", pack)
    pages = read(ROOT / cfg["qwen_run"] / "review-data.json")
    if pages["cohort_sha256"] != pack["cohort_sha256"]:
        raise ValueError("viewer_cohort_mismatch")
    by_id = {r["task_id"]: r for r in pages["cases"]}
    labels = {r["task_id"]: r for r in pack["reviews"]}
    cases = []
    for tid in failures:
        original = by_id[tid]
        if original["binding"] != labels[tid]["binding"] or original["status"] != "ok":
            raise ValueError("viewer_image_binding_mismatch")
        row = {k: original[k] for k in ("task_id", "index", "image_id", "prompt", "binding")}
        for role in ("source", "target"):
            asset = {}
            for kind in ("native", "preview"):
                source = (ROOT / cfg["qwen_run"] / original[role][kind]).resolve()
                if (ROOT / cfg["qwen_run"]).resolve() not in source.parents:
                    raise ValueError("image_outside_Qwen_viewer")
                dest = out / "media" / (sha256(source) + source.suffix)
                dest.parent.mkdir(exist_ok=True)
                if not dest.exists():
                    os.link(source, dest)
                asset[kind] = str(dest.relative_to(out))
            row[role] = asset
        cases.append(row)
    atomic(
        out / "review-data.json",
        {
            "schema": "qwen-label-recheck-v1",
            "raw_sha256": raw_sha,
            "cases": cases,
            "normalized": pack,
            "summary": summary,
        },
    )
    (out / "comparison.html").write_text(Path(__file__).with_name("viewer.html").read_text(), encoding="utf-8")
    atomic(out / "summary.json", summary)
    names = (
        "original-reviewed.json",
        "normalized-reviewed.json",
        "review-data.json",
        "comparison.html",
        "summary.json",
    )
    atomic(
        out / "receipt.json",
        {
            "raw_sha256": raw_sha,
            "code": code,
            "files_sha256": {n: sha256(out / n) for n in names},
            "final_correlation_report_run": False,
        },
    )
    if sha256(review_path) != raw_sha:
        raise ValueError("input_review_changed_during_prepare")
    return {"output": str(out), **summary}
