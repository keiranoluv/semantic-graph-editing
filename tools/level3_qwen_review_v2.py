"""Build a two-field Qwen review page from immutable inference outputs, on CPU."""

import argparse
import json
from pathlib import Path

from tools.level3_image_graph_v1.core import ROOT, digest, read, sha256, write
from tools.level3_qwen21_v1.core import bundle, finish
from tools.level3_qwen100_v1.viewer import thumb

FIELDS = ("edit_fulfillment", "preservation")
OUTPUT_ROOT = ROOT / "runs/level3/qwen-review-v2"


def build(run, output, review=None):
    run, output = Path(run).resolve(), Path(output).resolve()
    if OUTPUT_ROOT.resolve() not in output.parents:
        raise ValueError("output_must_be_under_qwen_review_v2")
    rows = bundle(run, "results.json")
    if not rows or len({r["task_id"] for r in rows}) != len(rows):
        raise ValueError("nonempty_unique_tasks_required")
    checksum = digest(rows)
    prior = {}
    if review:
        saved = read(review)
        if saved["results_sha256"] != checksum:
            raise ValueError("review_results_mismatch")
        for r in saved["reviews"]:
            if r.get("model", "qwen21") != "qwen21":
                continue
            if r["task_id"] in prior or r["task_id"] not in {x["task_id"] for x in rows}:
                raise ValueError("duplicate_or_unknown_review_task")
            if any(r.get(k) is not None and type(r[k]) is not bool for k in FIELDS):
                raise ValueError("review_labels_must_be_bool_or_null")
            prior[r["task_id"]] = r
    cases, reviews = [], []
    for row in rows:
        target = row.get("qwen21_path") or row.get("target_path")
        available = row["status"] == "ok" and bool(target)
        labels = {k: prior.get(row["task_id"], {}).get(k) for k in FIELDS}
        if not available and any(v is not None for v in labels.values()):
            raise ValueError("cannot_review_missing_target")
        reviews.append(
            {
                "task_id": row["task_id"],
                "model": "qwen21",
                "status": "reviewed" if all(type(v) is bool for v in labels.values()) else "pending",
                **labels,
            }
        )
        cases.append(
            {
                "task_id": row["task_id"],
                "prompt": row["prompt"],
                "source": thumb(row["source_path"]),
                "target": thumb(target) if available else None,
            }
        )
    template = {
        "schema": "qwen-only-review-v2",
        "results_sha256": checksum,
        "source_receipt_sha256": sha256(run / "receipt.json"),
        "review_policy": "two_manual_fields; no_notes_or_named_reviewer_required",
        "reviews": reviews,
    }
    download = "qwen21-reviewed-v2.json" if "qwen21_path" in rows[0] else "qwen100-reviewed-v2.json"
    data = json.dumps({"cases": cases, "review": template, "download": download}, ensure_ascii=False)
    for char, replacement in [("<", "\\u003c"), (">", "\\u003e"), ("&", "\\u0026")]:
        data = data.replace(char, replacement)
    source = Path(__file__).with_suffix(".html")
    html = source.read_text(encoding="utf-8").replace("__REVIEW_DATA__", data)
    output.mkdir(parents=True, exist_ok=False)
    write(output / "review-template.json", template)
    write(
        output / "provenance.json",
        {
            "input_run": str(run),
            "input_receipt_sha256": sha256(run / "receipt.json"),
            "prior_review": str(Path(review).resolve()) if review else None,
            "prior_review_sha256": sha256(review) if review else None,
            "prior_qwen_rows": len(prior),
            "code_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in [Path(__file__), source]},
            "new_inference": False,
            "native_outputs_modified": False,
        },
    )
    (output / "comparison.html").write_text(html, encoding="utf-8")
    finish(output)
    return output / "comparison.html"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--review", type=Path, help="Optional existing review bound to the same results hash")
    args = parser.parse_args()
    print(build(args.run, args.output, args.review))


if __name__ == "__main__":
    main()
