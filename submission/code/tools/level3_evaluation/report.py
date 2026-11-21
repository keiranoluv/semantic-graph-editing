"""Human labels enter only here, after automatic scores have been frozen."""
import csv
import html
import math
import os
from pathlib import Path

from scipy.stats import rankdata, spearmanr

from tools.level3_image_graph_v1.core import read, sha256
from tools.level3_qwen_unified.core import atomic

from .core import ROOT, case_header, read_stage, stage_binding

FIELDS = ("edit_fulfillment", "preservation")
METRICS = ("graph_consistency", "clipscore", "negative_lpips")


def validate_reviews(pack, records, cohort_sha256, allow_partial=False):
    if pack.get("schema") != "qwen-unified-review-v1" or pack.get("cohort_sha256") != cohort_sha256:
        raise ValueError("review_schema_or_cohort_mismatch")
    expected = {r["task"]["task_id"]: r["binding"] for r in records}
    labels, seen = {}, set()
    for row in pack["reviews"]:
        tid = row["task_id"]
        if tid not in expected or tid in seen:
            raise ValueError("unknown_or_duplicate_review_task:" + tid)
        seen.add(tid)
        if row.get("binding") != expected[tid]:
            raise ValueError("review_image_binding_mismatch:" + tid)
        if any(k not in row or (row[k] is not None and type(row[k]) is not bool) for k in FIELDS):
            raise ValueError("review_labels_must_be_boolean_or_null")
        complete = all(type(row[k]) is bool for k in FIELDS)
        if row.get("status") != ("reviewed" if complete else "pending"):
            raise ValueError("inconsistent_review_status")
        if complete:
            labels[tid] = {k: row[k] for k in FIELDS}
            labels[tid]["overall_pass"] = row["edit_fulfillment"] and row["preservation"]
    if not allow_partial and (seen != set(expected) or len(labels) != len(expected)):
        raise ValueError(
            f"need_all_{len(expected)}_human_reviews; complete={len(labels)}; use_--allow-partial_for_preliminary_report"
        )
    return labels


def correlation(rows, metric, label):
    available = [r for r in rows if r.get(metric) is not None and r.get(label) is not None]
    ids = [r["task_id"] for r in available]
    x, y = [r[metric] for r in available], [int(r[label]) for r in available]
    result = {"n": len(x), "task_ids": ids, "spearman": None, "reason": None}
    if len(x) < 2:
        result["reason"] = "fewer_than_two_cases"
    elif len(set(x)) < 2 or len(set(y)) < 2:
        result["reason"] = "constant_scores_or_labels"
    else:
        result["spearman"] = float(spearmanr(x, y).statistic)
    # No naive independent-sample p-value: two edits share each source image.
    return result


def comparison(rows):
    common = [r for r in rows if all(r[k] is not None for k in METRICS) and r.get("overall_pass") is not None]
    return {
        "total_pairs": len(rows),
        "human_labeled": sum(r.get("overall_pass") is not None for r in rows),
        "graph_available": sum(r["graph_consistency"] is not None for r in rows),
        "common_n": len(common),
        "common_case_correlations": {m: correlation(common, m, "overall_pass") for m in METRICS},
        "all_available_secondary_correlations": {m: correlation(rows, m, "overall_pass") for m in METRICS},
        "components": {
            "edit_vs_fulfillment": correlation(rows, "edit_score", "edit_fulfillment"),
            "preservation_vs_human": correlation(rows, "evaluable_preservation", "preservation"),
        },
    }


def disagreements(rows, limit=10):
    common = [r for r in rows if all(r[k] is not None for k in METRICS) and r.get("overall_pass") is not None]
    if len(common) < 2:
        return []
    ranks = {m: (rankdata([r[m] for r in common]) - 1) / (len(common) - 1) for m in METRICS}
    candidates = []
    for i, r in enumerate(common):
        gaps = {m: float(abs(ranks["graph_consistency"][i] - ranks[m][i])) for m in METRICS[1:]}
        candidates.append(
            {
                **r,
                "rank_percentiles": {m: float(ranks[m][i]) for m in METRICS},
                "rank_gaps": gaps,
                "maximum_rank_gap": max(gaps.values()),
                "qualitative_analysis": None,
                "status": "candidate_needs_visual_inspection",
            }
        )
    return sorted(candidates, key=lambda r: (-r["maximum_rank_gap"], r["task_id"]))[:limit]


def validate_stage(pack, records, binding):
    if pack.get("evaluation_eligible") is not True or pack.get("stage_binding") != binding:
        raise ValueError("wrong_stage_protocol_or_smoke_not_eligible")
    if [r["task_id"] for r in pack["rows"]] != [r["task"]["task_id"] for r in records]:
        raise ValueError("stage_task_membership_changed")
    for row, record in zip(pack["rows"], records, strict=True):
        if any(row.get(k) != v for k, v in case_header(record).items()):
            raise ValueError("stage_case_binding_mismatch")


def run(out, cfg, records, review_path, allow_partial=False):
    review_path = Path(review_path)
    labels = validate_reviews(read(review_path), records, sha256(ROOT / cfg["cohort"]), allow_partial)
    binding = stage_binding(cfg, records)
    graph = read_stage(out / "graph/results.json")
    metrics = read_stage(out / "metrics/results.json")
    for pack in (graph, metrics):
        validate_stage(pack, records, binding)
    rows = []
    for g, m in zip(graph["rows"], metrics["rows"], strict=True):
        s = g["scores"]
        values = {k: m[k] for k in ("clipscore", "lpips", "negative_lpips")}
        values.update({k: s[k] for k in ("edit_score", "preservation_score", "graph_consistency")})
        if any(v is not None and (type(v) not in (float, int) or not math.isfinite(v)) for v in values.values()):
            raise ValueError("nonfinite_or_non_numeric_score")
        rows.append(
            {
                **{k: g[k] for k in ("task_id", "image_id", "dataset", "operation")},
                **values,
                "evaluable_preservation": s["preservation_score"] if s["preservation_evaluable"] else None,
                "reason_codes": s["reason_codes"],
                **labels.get(g["task_id"], {}),
            }
        )
    summary = {
        "schema": "level3-evaluation-report-v1",
        "preliminary": len(labels) != len(records),
        "human_labels_used_for_tuning": False,
        "review_sha256": sha256(review_path),
        "graph_sha256": sha256(out / "graph/results.json"),
        "metrics_sha256": sha256(out / "metrics/results.json"),
        "protocol": cfg["evaluation_protocol"],
        "all": comparison(rows),
        "by_operation": {
            k: comparison([r for r in rows if r["operation"] == k]) for k in sorted({r["operation"] for r in rows})
        },
        "by_dataset": {
            k: comparison([r for r in rows if r["dataset"] == k]) for k in sorted({r["dataset"] for r in rows})
        },
    }
    # Every label version gets its own report; existing reports are never overwritten.
    dest = out / "reports" / sha256(review_path)[:16]
    if dest.exists():
        raise ValueError("report_already_exists:" + str(dest))
    dest.mkdir(parents=True)
    atomic(dest / "summary.json", summary)
    atomic(dest / "labels.json", read(review_path))
    atomic(dest / "results.json", rows)
    candidates = disagreements(rows)
    atomic(dest / "disagreement-candidates.json", candidates)
    fields = [
        "task_id",
        "image_id",
        "dataset",
        "operation",
        "edit_score",
        "preservation_score",
        "evaluable_preservation",
        "graph_consistency",
        "clipscore",
        "lpips",
        "negative_lpips",
        *FIELDS,
        "overall_pass",
    ]
    with (dest / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    by_id = {r["task"]["task_id"]: r for r in records}
    cards = []
    for candidate in candidates:
        record = by_id[candidate["task_id"]]
        images = []
        for role in ("source", "target"):
            path = Path(record["image"][role + "_path"])
            asset = dest / "media" / (sha256(path) + path.suffix)
            asset.parent.mkdir(exist_ok=True)
            if not asset.exists():
                os.link(path, asset)
            images.append(f'<img alt="{role}" src="media/{asset.name}">')
        cards.append(
            "<article><h2>"
            + html.escape(candidate["task_id"])
            + "</h2><p>"
            + html.escape(record["task"]["instruction_en"])
            + "</p><div>"
            + "".join(images)
            + "</div><pre>"
            + html.escape(str({k: candidate[k] for k in (*METRICS, *FIELDS)}))
            + "</pre><p>Ứng viên bất đồng; chưa có phân tích định tính.</p></article>"
        )
    table = "".join(
        f'<tr><td>{m}</td><td>{v["n"]}</td><td>{v["spearman"]}</td><td>{v["reason"] or ""}</td></tr>'
        for m, v in summary["all"]["common_case_correlations"].items()
    )
    (dest / "comparison.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Level 3 evaluation</title>'
        "<style>body{font:16px system-ui;max-width:1400px;margin:30px auto}img{width:49%;vertical-align:top}td{padding:10px}article{border-top:1px solid #ccc;margin-top:30px}</style>"
        "<h1>Đánh giá Level 3</h1><p>Spearman trên cùng các mẫu hợp lệ; None là chưa xác định.</p><table>"
        + table
        + "</table>"
        + "".join(cards),
        encoding="utf-8",
    )
    return {
        "output": str(dest),
        "labeled": len(labels),
        "common_cases": summary["all"]["common_n"],
        "disagreement_candidates": len(candidates),
        "preliminary": summary["preliminary"],
    }
