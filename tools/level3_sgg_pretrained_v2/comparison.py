"""Local comparison with images, boxes, directed graphs and manual review packet."""

import html
import json
import math

from tools.level3_image_graph_v1.core import write
from tools.level3_sgg_compare_v1.comparison import render_comparison, statistics

from .evaluation import MODELS, evaluate, template


def make_comparison(rows, models):
    expected = {r["image_id"]: r for r in rows}
    if len(expected) != 5 or len(rows) != 5:
        raise ValueError("exact_five_required")
    indexed = {}
    for model in MODELS:
        records = models.get(model, [])
        indexed[model] = {r["candidate_id"]: r for r in records}
        if len(indexed[model]) != len(records) or not set(indexed[model]) <= set(expected):
            raise ValueError("unexpected_or_duplicate_image")
        for cid, r in indexed[model].items():
            if r["image"] != expected[cid] or r["source_sha256"] != expected[cid]["sha256"]:
                raise ValueError("image_identity_mismatch")
    cases = []
    for cid, row in expected.items():
        items = {}
        for m in MODELS:
            r = indexed[m].get(
                cid, {"candidate_id": cid, "image": row, "source_sha256": row["sha256"], "status": "not_run"}
            )
            items[m] = {"record": r, "statistics": statistics(r)}
        cases.append({"candidate_id": cid, "image": row, "models": items})
    return {
        "schema": "level3-sgg-comparison-v2",
        "cases": cases,
        "image_count": 5,
        "quality_conclusion": "requires_human_review",
        "source_target_stability": "not_measured",
    }


def graph_svg(graph, prefix):
    nodes, edges = graph["nodes"], graph["edges"]
    if not nodes:
        return "<p>Không có graph khả dụng.</p>"
    height = max(280, len(nodes) * 16)
    positions = {
        n["id"]: (
            240 + 180 * math.cos(2 * math.pi * i / len(nodes)),
            height / 2 + (height / 2 - 40) * math.sin(2 * math.pi * i / len(nodes)),
        )
        for i, n in enumerate(nodes)
    }
    parts = [
        f'<svg viewBox="0 0 480 {height}" role="img" aria-label="Directed graph"><defs><marker id="{prefix}" markerWidth="7" markerHeight="7" refX="16" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3z" fill="#51748a"/></marker></defs>'
    ]
    for edge in edges:
        x1, y1 = positions[edge["subject"]]
        x2, y2 = positions[edge["object"]]
        title = html.escape(f"{edge['subject']} → {edge['predicate']} → {edge['object']}")
        parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#51748a" opacity=".5" marker-end="url(#{prefix})"><title>{title}</title></line>'
        )
    for n in nodes:
        x, y = positions[n["id"]]
        parts.append(
            f'<g><title>{html.escape(n["label"])}</title><circle cx="{x}" cy="{y}" r="15" fill="#d9eef7"/><text x="{x}" y="{y + 4}" text-anchor="middle" font-size="10">{html.escape(n["id"])}</text></g>'
        )
    return "".join(parts) + "</svg>"


def export(rows, models, locked, output):
    comparison = make_comparison(rows, models)
    write(output / "comparison.json", comparison)
    write(output / "review-template.json", template(comparison))
    write(output / "metrics.json", evaluate(comparison))
    # Reuse the audited local image embedding/box selector without changing v1 source.
    path = output / "comparison.html"
    render_comparison(comparison, locked, path)
    page = path.read_text()
    page = page.replace("grid-template-columns:1fr 1fr;", "grid-template-columns:repeat(3,minmax(0,1fr));")
    start, end = page.index("<h1>"), page.index("</p>", page.index("<h1>")) + 4
    page = (
        page[:start]
        + (
            "<h1>ROBIN-SAM · REACT++ · RelateAnything + OWLv2</h1><p>"
            "Cùng 5 ảnh đã khóa. Smoke chỉ chạy ảnh đầu; not_run không phải graph rỗng. "
            "Số node/cạnh và confidence không chứng minh độ chính xác. "
            "Hai mô hình mới không dự đoán attribute. Duyệt review-template.json rồi dùng lệnh evaluate; "
            "chỉ xác nhận inventory_complete sau khi liệt kê đầy đủ phần bị bỏ sót. "
            "Rê chuột lên đỉnh/cạnh để xem nhãn; xem danh sách bên dưới để đọc đầy đủ quan hệ có hướng.</p>"
        )
        + page[end:]
    )
    marker = "<h4>Đối tượng / thuộc tính</h4>"
    pieces = page.split(marker)
    rebuilt = pieces[0]
    i = 0
    for case in comparison["cases"]:
        for item in case["models"].values():
            graph = item["record"].get("source_graph") or {"nodes": [], "edges": []}
            evidence = {
                "status": item["record"].get("status"),
                "error": item["record"].get("error"),
                "grounding_and_confidence": item["record"].get("grounding"),
            }
            panel = "<details><summary>Confidence / grounding / lỗi</summary><pre>"
            panel += html.escape(json.dumps(evidence, ensure_ascii=False, indent=2)) + "</pre></details>"
            rebuilt += graph_svg(graph, f"arrow{i}") + panel + marker + pieces[i + 1]
            i += 1
    path.write_text(rebuilt)
    return comparison
