"""Local offline viewer and hash-bound, append-only human correction import."""

import base64
import copy
import html
import io
import json
import math
from pathlib import Path

from PIL import Image

from src.editing.operations import diff_aligned_graphs

from .core import digest, parse_graph, sha256, validate_box, validate_record, write

AXES = ("objects", "identities_boxes", "attributes", "directed_relations")


def graph_svg(graph):
    count = max(1, len(graph.nodes))
    width, height = 900, max(500, count * 22)
    positions = {
        n.id: (
            width / 2 + width * 0.35 * math.cos(2 * math.pi * i / count),
            height / 2 + height * 0.38 * math.sin(2 * math.pi * i / count),
        )
        for i, n in enumerate(graph.nodes)
    }
    esc = html.escape
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="#64748b"/></marker></defs>',
    ]
    for e in graph.edges:
        x1, y1 = positions[e.subject]
        x2, y2 = positions[e.object]
        parts += [
            f'<path d="M{x1},{y1} Q{(x1+x2)/2+20},{(y1+y2)/2-20} {x2},{y2}" stroke="#64748b" fill="none" marker-end="url(#arrow)"/>',
            f'<text x="{(x1+x2)/2+10}" y="{(y1+y2)/2-10}" font-size="12">{esc(e.id+": "+e.predicate)}</text>',
        ]
    for n in graph.nodes:
        x, y = positions[n.id]
        parts += [
            f'<circle cx="{x}" cy="{y}" r="8" fill="#0f766e"/>',
            f'<text x="{x+10}" y="{y}" font-size="14">{esc(n.id+": "+n.label)}</text>',
            f'<text x="{x+10}" y="{y+17}" font-size="11">{esc(", ".join(n.attributes))}</text>',
        ]
    return "".join(parts) + "</svg>"


def render_review(output, records, locked):
    output = Path(output)
    pages = []
    for record in records:
        graph = validate_record(record)
        row = record["image"]
        source = Path(locked) / row["image_path"]
        if sha256(source) != row["sha256"]:
            raise ValueError("review_image_changed")
        with Image.open(source) as image:
            image = image.convert("RGB")  # stored-pixel orientation matches model coordinates
            image.thumbnail((1000, 1000))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=85)
        thumbnail = base64.b64encode(buffer.getvalue()).decode()
        identifier = record["candidate_id"]
        svg = graph_svg(graph)
        (output / f"{identifier}.svg").write_text(svg, encoding="utf-8")
        pages.append(
            {
                "record": record,
                "thumbnail": thumbnail,
                "graph_svg": svg,
                "submission": {
                    "candidate_id": identifier,
                    "base_record_sha256": digest(record),
                    "decision": "pending",
                    "reviewer": "",
                    "notes": "",
                    "checks": {a: False for a in AXES},
                    "source_graph": record["source_graph"],
                    "boxes": {n.id: record["grounding"]["nodes"][n.id]["bbox_xyxy"] for n in graph.nodes},
                },
            }
        )
    data = json.dumps(pages, ensure_ascii=False).replace("<", "\\u003c").replace("&", "\\u0026")
    template = Path(__file__).with_name("viewer.html").read_text(encoding="utf-8")
    (output / "review.html").write_text(template.replace("__REVIEW_DATA__", data), encoding="utf-8")
    write(
        output / "review-template.json",
        {"schema": "level3-image-graph-review-v1", "reviews": [p["submission"] for p in pages]},
    )


def apply_review(records, submission):
    if submission["schema"] != "level3-image-graph-review-v1":
        raise ValueError("unknown_review_schema")
    originals = {r["candidate_id"]: r for r in records}
    rows = submission["reviews"]
    ids = [r["candidate_id"] for r in rows]
    if len(set(ids)) != len(ids) or set(ids) != set(originals):
        raise ValueError("review_ids_must_cover_exact_candidate_bundle")
    accepted, decisions = [], []
    for review in rows:
        original = originals[review["candidate_id"]]
        if review["base_record_sha256"] != digest(original):
            raise ValueError("stale_review_evidence")
        decision = review["decision"]
        if decision not in {"pending", "accept", "reject"}:
            raise ValueError("invalid_review_decision")
        if decision != "pending" and (not review["reviewer"].strip() or not review["notes"].strip()):
            raise ValueError("reviewer_and_notes_required")
        decisions.append({k: review[k] for k in ("candidate_id", "decision", "reviewer", "notes")})
        if decision != "accept":
            continue
        if set(review["checks"]) != set(AXES) or any(review["checks"][a] is not True for a in AXES):
            raise ValueError("all_review_axes_must_be_checked")
        old = validate_record(original)
        graph = parse_graph(review["source_graph"])
        if not graph.nodes:
            raise ValueError("empty_graph_cannot_be_accepted")
        boxes = review["boxes"]
        if set(boxes) != {n.id for n in graph.nodes}:
            raise ValueError("review_boxes_must_match_nodes")
        record = copy.deepcopy(original)
        old_nodes, old_edges = {n.id: n for n in old.nodes}, {e.id: e for e in old.edges}
        grounding = {"nodes": {}, "edges": {}}
        for n in graph.nodes:
            box = validate_box(boxes[n.id], original["image"]["width"], original["image"]["height"])
            previous = original["grounding"]["nodes"].get(n.id)
            if old_nodes.get(n.id) == n and previous and box == previous["bbox_xyxy"]:
                grounding["nodes"][n.id] = copy.deepcopy(previous)
            else:
                grounding["nodes"][n.id] = {
                    "bbox_xyxy": box,
                    "confidence": None,
                    "origin": "human_correction",
                    "confidence_kind": "not_scored",
                    "raw_detection_ids": previous["raw_detection_ids"] if previous else [],
                    "attributes": [
                        {"label": a, "confidence": None, "origin": "human_correction", "confidence_kind": "not_scored"}
                        for a in n.attributes
                    ],
                }
        for e in graph.edges:
            grounding["edges"][e.id] = (
                copy.deepcopy(original["grounding"]["edges"][e.id])
                if old_edges.get(e.id) == e
                and grounding["nodes"][e.subject]["origin"] != "human_correction"
                and grounding["nodes"][e.object]["origin"] != "human_correction"
                else {
                    "confidence": None,
                    "origin": "human_correction",
                    "confidence_kind": "not_scored",
                    "raw_relation_ids": [],
                }
            )
        record.update(
            source_graph=graph.to_dict(), grounding=grounding, annotation_status="human_accepted", human_reviewed=True
        )
        record["review"] = {
            **review,
            "operations_from_candidate": [o.to_dict() for o in diff_aligned_graphs(old, graph)],
            "original_record_sha256": digest(original),
            "review_sha256": digest(review),
        }
        validate_record(record)
        accepted.append(record)
    return accepted, decisions
