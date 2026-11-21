"""Portable notebook helpers. Read saved predictions; recompute graph scores on CPU."""
import base64
import html
import json
import math
from pathlib import Path

from task3_vn100.submission.bundle import DEFAULT, read, safe_name, sha


def open_bundle(root, manifest=None):
    root = Path(root).resolve()
    if manifest is None:
        manifest = root / "portable-manifest.json"
        if not manifest.exists():
            manifest = root / DEFAULT / "portable-manifest.json"
    data = read(manifest)
    if data["schema"] != "level3-portable-evidence-v1":
        raise ValueError("wrong_manifest_schema")
    return root, data


def checked(root, manifest, name):
    path = root / safe_name(name)
    if sha(path) != manifest["files"][name]["sha256"]:
        raise ValueError("demo_artifact_changed:" + name)
    return path


def recompute(root, manifest, task_id, images=True):
    from task3_vn100.evaluation.scoring import score

    case = next(c for c in manifest["cases"] if c["task_id"] == task_id)
    graphs = {r: read(checked(root, manifest, p)) for r, p in case["graphs"].items()}
    if images:
        for role in ("source", "target"):
            checked(root, manifest, case[role])
    result = score({"task": case["task"], "author": case["author"], **graphs}, manifest["graph_config"])
    reference = next(
        r["scores"] for r in read(checked(root, manifest, manifest["graph_scores"]))["rows"] if r["task_id"] == task_id
    )
    if json.dumps(result, sort_keys=True) != json.dumps(reference, sort_keys=True):
        raise ValueError("recomputed_score_differs_from_frozen_result:" + task_id)
    return {**case, "graphs_loaded": graphs, "recomputed": result}


def verify_all_scores(root, manifest):
    from task3_vn100.evaluation.report import comparison

    rows = []
    for case in manifest["cases"]:
        result = recompute(root, manifest, case["task_id"], images=False)["recomputed"]
        row = dict(case["scores_and_labels"])
        row.update({k: result[k] for k in ("edit_score", "preservation_score", "graph_consistency")})
        row["evaluable_preservation"] = result["preservation_score"] if result["preservation_evaluable"] else None
        rows.append(row)
    actual = comparison(rows)
    expected = read(checked(root, manifest, manifest["summary"]))["all"]
    if actual != expected:
        raise ValueError("recomputed_correlations_differ")
    return {
        "pairs_recomputed": len(rows),
        "graph_available": actual["graph_available"],
        "common_n": actual["common_n"],
        "spearman": {m: r["spearman"] for m, r in actual["common_case_correlations"].items()},
    }


def table(headers, rows):
    esc = html.escape
    return (
        '<table style="border-collapse:collapse;width:100%"><tr>'
        + "".join(
            f'<th style="text-align:left;padding:6px;border-bottom:1px solid #aaa">{esc(str(h))}</th>' for h in headers
        )
        + "</tr>"
        + "".join(
            "<tr>"
            + "".join(f'<td style="padding:6px;border-bottom:1px solid #ddd">{esc(str(c))}</td>' for c in row)
            + "</tr>"
            for row in rows
        )
        + "</table>"
    )


def image_pair_html(root, case, boxes=True):
    from task3_vn100.analysis.core import graph_svg

    blocks = []
    for role, title in (("source", "Ảnh nguồn"), ("target", "Ảnh Qwen")):
        path = root / case[role]
        payload = base64.b64encode(path.read_bytes()).decode()
        mime = "image/png" if path.suffix == ".png" else "image/jpeg"
        overlay = (
            graph_svg(case["graphs_loaded"][role], case["author"]["grounding"]["region"] if role == "target" else None)
            if boxes
            else ""
        )
        blocks.append(
            f'<div style="flex:1;min-width:280px"><b>{title}</b><div style="position:relative"><img style="display:block;width:100%" src="data:{mime};base64,{payload}">{overlay}</div></div>'
        )
    style = "<style>.l3-demo .boxes{position:absolute;inset:0;width:100%;height:100%;fill:none;stroke:#e44232;stroke-width:3}.l3-demo .boxes text{fill:white;stroke:black;stroke-width:1;paint-order:stroke}.l3-demo .roi{stroke:#ffdf00;stroke-width:6;stroke-dasharray:14 8}</style>"
    return style + '<div class="l3-demo" style="display:flex;flex-wrap:wrap;gap:15px">' + "".join(blocks) + "</div>"


def graph_html(graph, title, marker="arrow"):
    graph = graph.get("source_graph", graph)
    nodes, edges = graph["nodes"], graph["edges"]
    pos = {
        n["id"]: (
            270 + 190 * math.cos(2 * math.pi * i / max(len(nodes), 1)),
            250 + 170 * math.sin(2 * math.pi * i / max(len(nodes), 1)),
        )
        for i, n in enumerate(nodes)
    }
    svg = f'<svg viewBox="0 0 540 500" style="width:100%;max-height:480px"><defs><marker id="{marker}" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#536979"/></marker></defs>'
    for edge in edges:
        x, y = pos[edge["subject"]]
        u, v = pos[edge["object"]]
        dx, dy = u - x, v - y
        length = max(math.hypot(dx, dy), 1)
        x, y, u, v = x + dx / length * 32, y + dy / length * 32, u - dx / length * 35, v - dy / length * 35
        cx, cy = (x + u) / 2 - dy / length * 23, (y + v) / 2 + dx / length * 23
        svg += f'<path d="M{x},{y} Q{cx},{cy} {u},{v}" fill="none" stroke="#536979" marker-end="url(#{marker})"><title>{html.escape(edge["predicate"])}</title></path>'
    for node in nodes:
        x, y = pos[node["id"]]
        svg += f'<circle cx="{x}" cy="{y}" r="33" fill="#e5f1ff" stroke="#24669a"/><text x="{x}" y="{y-4}" text-anchor="middle" font-size="12">{html.escape(node["id"])}</text><text x="{x}" y="{y+12}" text-anchor="middle" font-size="12">{html.escape(node["label"])}</text>'
    labels = {n["id"]: n["label"] for n in nodes}
    rows = [
        [f'{e["subject"]}: {labels[e["subject"]]}', e["predicate"], f'{e["object"]}: {labels[e["object"]]}']
        for e in edges
    ]
    return f"<h4>{html.escape(title)}</h4>" + svg + "</svg>" + table(["Subject →", "Quan hệ", "→ Object"], rows)


def task2_demo(root, manifest):
    from common.editing.operations import EditGraph, Operation, apply_operations

    cases = read(checked(root, manifest, manifest["task2_cases"]))
    result = read(checked(root, manifest, manifest["task2_result"]))
    selected = [next(c for c in cases if c["correct"]), next(c for c in cases if not c["correct"] and not c["error"])]
    outputs = []
    for case in selected:
        predicted = apply_operations(
            EditGraph.from_dict(case["source"]), [Operation.from_dict(o) for o in case["predicted"]]
        )
        outputs.append(
            {**case, "predicted_target": predicted.to_dict(), "checkpoint_sha256": result["checkpoint_sha256"]}
        )
    return outputs
