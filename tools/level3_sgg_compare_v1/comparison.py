"""Same-image descriptive comparison and hash-bound human quality rubric."""

import base64
import html
import io
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from tools.level3_image_graph_v1.core import digest, sha256, validate_record, write

AXES = ("objects", "attributes", "directed_relations", "grounding", "overall")


def statistics(record):
    if record.get("status") != "ok":
        return {
            "status": record.get("status", "missing"),
            "nodes": None,
            "attributes": None,
            "edges": None,
            "grounded_nodes": None,
            "human_quality": None,
        }
    graph = validate_record(record)
    return {
        "status": "ok",
        "nodes": len(graph.nodes),
        "attributes": sum(len(n.attributes) for n in graph.nodes),
        "edges": len(graph.edges),
        "grounded_nodes": len(record["grounding"]["nodes"]),
        "isolated_nodes": sum(not any(n.id in (e.subject, e.object) for e in graph.edges) for n in graph.nodes),
        "self_relations": sum(e.subject == e.object for e in graph.edges),
        "human_quality": None,
    }


def make_comparison(rows, baseline, robin):
    if len(rows) != 5 or len({r["image_id"] for r in rows}) != 5:
        raise ValueError("comparison_requires_exact_same_five_images")
    models = {
        "reltr-clip": {r["candidate_id"]: r for r in baseline},
        "robin-sam": {r["candidate_id"]: r for r in robin},
    }
    expected = {r["image_id"] for r in rows}
    for model, records in models.items():
        source = baseline if model == "reltr-clip" else robin
        if len(records) != len(source) or (set(records) != expected and not (model == "robin-sam" and not records)):
            raise ValueError("comparison_requires_exact_same_five_images")
        for row in rows:
            record = records.get(row["image_id"])
            if record is not None and (record["source_sha256"] != row["sha256"] or record["image"] != row):
                raise ValueError("comparison_image_identity_mismatch")
    output = {
        "schema": "level3-sgg-comparison-v1",
        "image_count": len(rows),
        "cases": [],
        "quality_conclusion": "not_measured_without_human_review",
        "notes": [
            "Counts and graph density are not accuracy metrics.",
            "RelTR predicts boxes; ROBIN describes SAM input regions: grounding sources differ.",
            "ROBIN labels are full region descriptions; exact label-string matching is not a fair quality score.",
            "Attribute projection is partial; inspect the native description as well.",
        ],
    }
    reviews = []
    for row in rows:
        case = {"candidate_id": row["image_id"], "image": row, "models": {}}
        for model, records in models.items():
            record = records.get(
                row["image_id"], {"candidate_id": row["image_id"], "source_sha256": row["sha256"], "status": "not_run"}
            )
            case["models"][model] = {"record": record, "statistics": statistics(record)}
            reviews.append(
                {
                    "candidate_id": row["image_id"],
                    "model": model,
                    "record_sha256": digest(record),
                    "status": "pending",
                    "reviewer": "",
                    "notes": "",
                    "scores": {axis: None for axis in AXES},
                }
            )
        output["cases"].append(case)
    rubric = {
        "schema": "level3-sgg-quality-review-v1",
        "comparison_sha256": digest(output),
        "rubric": {
            "0": "mostly incorrect / unusable",
            "1": "mixed; substantial corrections needed",
            "2": "mostly correct / usable",
            "objects": "presence, description/category and instance identity; count missing salient objects in notes",
            "attributes": "ownership and visible correctness; inspect native descriptions; null if unassessable",
            "directed_relations": "predicate correctness, endpoint identity and direction; note missing meaningful relations",
            "grounding": "region/box actually covers the described instance; SAM IoU confidence is not label accuracy",
            "overall": "usable for edit correctness AND unrelated-scene preservation; no performance claims from count alone",
        },
        "reviews": reviews,
    }
    return output, rubric


def validate_quality(comparison, submitted):
    if submitted["schema"] != "level3-sgg-quality-review-v1" or submitted["comparison_sha256"] != digest(comparison):
        raise ValueError("stale_quality_review")
    expected = {
        (case["candidate_id"], model): data["record"]
        for case in comparison["cases"]
        for model, data in case["models"].items()
    }
    rows = submitted["reviews"]
    keys = [(r["candidate_id"], r["model"]) for r in rows]
    if len(set(keys)) != len(keys) or set(keys) != set(expected):
        raise ValueError("quality_review_must_cover_all_image_model_pairs")
    summaries = {}
    for model in ("reltr-clip", "robin-sam"):
        summaries[model] = {"reviewed": 0, "scores": {a: [] for a in AXES}}
    for row in rows:
        record = expected[row["candidate_id"], row["model"]]
        if row["record_sha256"] != digest(record) or row["status"] not in {"pending", "reviewed"}:
            raise ValueError("quality_review_evidence_mismatch")
        if row["status"] == "pending":
            continue
        if record.get("status") != "ok" or not row["reviewer"].strip() or not row["notes"].strip():
            raise ValueError("cannot_score_missing_prediction_or_anonymous_review")
        if set(row["scores"]) != set(AXES):
            raise ValueError("incomplete_quality_axes")
        for axis, value in row["scores"].items():
            if value is not None and (type(value) is not int or value not in (0, 1, 2)):
                raise ValueError("quality_score_must_be_0_1_2_or_null")
            if value is None and axis != "attributes":
                raise ValueError("only_attribute_axis_may_be_unassessable")
            if value is not None:
                summaries[row["model"]]["scores"][axis].append(value)
        summaries[row["model"]]["reviewed"] += 1
    return {
        "schema": "level3-sgg-quality-summary-v1",
        "comparison_sha256": digest(comparison),
        "submission_sha256": digest(submitted),
        "models": summaries,
        "complete_paired_review": all(s["reviewed"] == len(comparison["cases"]) for s in summaries.values()),
        "interpretation": "ordinal diagnostic ratings on five development images, not benchmark accuracy",
    }


def render_comparison(comparison, locked, output):
    """Standalone local page: original photos, per-model boxes and directed graph lists."""
    parts = [
        '<!doctype html><html lang="vi"><meta charset="utf-8"><title>SGG pilot 5</title>',
        "<style>body{font:16px system-ui;margin:24px;color:#143247;background:#f2f6fa}article{background:white;padding:18px;margin:20px 0;border-radius:12px}.columns{display:grid;grid-template-columns:1fr 1fr;gap:20px}svg{width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere}li{margin:6px}select{max-width:100%;padding:8px}.muted{color:#835000}@media(max-width:900px){.columns{display:block}}</style>",
        "<h1>So sánh SGG · cùng 5 ảnh pilot</h1><p>Đây là tập phát triển. Số node/cạnh không phải chất lượng. ROBIN dùng mask SAM đầu vào; confidence SAM không phải confidence nhãn ROBIN. Chấm 5 tiêu chí trong quality-review-template.json rồi nhập bằng CLI.</p>",
    ]
    for case in comparison["cases"]:
        row = case["image"]
        path = Path(locked) / row["image_path"]
        if sha256(path) != row["sha256"]:
            raise ValueError("comparison_view_image_changed")
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((1200, 1200))
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=90)
        data = base64.b64encode(buf.getvalue()).decode()
        parts.append("<article><h2>" + html.escape(case["candidate_id"]) + '</h2><div class="columns">')
        for model, result in case["models"].items():
            record = result["record"]
            graph = record.get("source_graph") or {"nodes": [], "edges": []}
            grounding = record.get("grounding", {"nodes": {}})
            parts.append(
                "<section><h3>"
                + html.escape(model)
                + '</h3><p class="muted">'
                + html.escape(record.get("status", "missing"))
                + "</p>"
            )
            parts.append(
                "<select onchange=\"const s=this.parentElement.querySelector('svg');s.querySelectorAll('[data-node]').forEach(e=>e.style.display=this.value==='all'||e.dataset.node===this.value?'':'none')\"><option value=\"none\">Ảnh không có hộp</option><option value=\"all\">Tất cả hộp</option>"
            )
            for n in graph["nodes"]:
                parts.append(
                    f'<option value="{html.escape(n["id"], quote=True)}">{html.escape(n["id"] + ": " + n["label"])}</option>'
                )
            parts.append(
                f'</select><svg viewBox="0 0 {row["width"]} {row["height"]}"><image href="data:image/jpeg;base64,{data}" width="{row["width"]}" height="{row["height"]}"/>'
            )
            for n in graph["nodes"]:
                x1, y1, x2, y2 = grounding["nodes"][n["id"]]["bbox_xyxy"]
                identifier = html.escape(n["id"], quote=True)
                parts.append(
                    f'<g data-node="{identifier}" style="display:none"><rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" fill="none" stroke="#ffe000" stroke-width="4"/><text x="{x1}" y="{max(24,y1)}" font-size="28" fill="white" stroke="black" paint-order="stroke">{identifier}</text></g>'
                )
            parts.append(
                "</svg><pre>"
                + html.escape(json.dumps(result["statistics"], ensure_ascii=False))
                + "</pre><h4>Đối tượng / thuộc tính</h4><ul>"
            )
            for n in graph["nodes"]:
                parts.append("<li>" + html.escape(f'{n["id"]}: {n["label"]} | attributes={n["attributes"]}') + "</li>")
            parts.append("</ul><h4>Quan hệ có hướng</h4><ul>")
            for edge in graph["edges"]:
                parts.append(
                    "<li>" + html.escape(f'{edge["subject"]} — {edge["predicate"]} → {edge["object"]}') + "</li>"
                )
            parts.append(
                "</ul><details><summary>Native output / provenance</summary><pre>"
                + html.escape(
                    json.dumps(
                        {"native": record.get("native_output"), "provenance": record.get("provenance")},
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                + "</pre></details></section>"
            )
        parts.append("</div></article>")
    Path(output).write_text("".join(parts) + "</html>", encoding="utf-8")


def export_comparison(rows, baseline, robin, locked, output):
    comparison, rubric = make_comparison(rows, baseline, robin)
    write(Path(output) / "comparison.json", comparison)
    write(Path(output) / "quality-review-template.json", rubric)
    render_comparison(comparison, locked, Path(output) / "comparison.html")
    return {
        model: dict(Counter(c["models"][model]["statistics"]["status"] for c in comparison["cases"]))
        for model in ("reltr-clip", "robin-sam")
    }
