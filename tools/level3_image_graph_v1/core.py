"""CPU contracts: immutable pilot inputs, EditGraph sidecars and postprocessing."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

from PIL import Image

from src.editing.operations import Edge, EditGraph, Node

from . import VERSION

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCKED = ROOT / "data/processed/level3/vietnam-benchmark-v2/pilot/locked"
DEFAULT_CONFIG = ROOT / "configs/level3/image-graph-v1.json"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def outside_locked(path, locked):
    path, locked = Path(path).resolve(), Path(locked).resolve()
    if path == locked or locked in path.parents or path in locked.parents:
        raise ValueError("output_must_be_separate_from_locked_dataset")
    return path


def load_pilot(locked):
    locked = Path(locked).resolve()
    manifest = read(locked / "manifest.json")
    rows = manifest["images"]
    tasks = [json.loads(x) for x in (locked / "source_graph_tasks.jsonl").read_text().splitlines() if x.strip()]
    ids = [r["image_id"] for r in rows]
    if len(rows) != 30 or manifest["image_count"] != 30 or len(set(ids)) != 30:
        raise ValueError("expected_30_unique_pilot_images")
    if Counter(r["dataset"] for r in rows) != {"sidewalk": 15, "travel": 15}:
        raise ValueError("pilot_quota_mismatch")
    if len(tasks) != 30 or {t["image_id"] for t in tasks} != set(ids):
        raise ValueError("pilot_tasks_mismatch")
    task_by_id = {t["image_id"]: t for t in tasks}
    for row in rows:
        if not re.fullmatch(r"[a-z0-9-]+", row["image_id"]):
            raise ValueError("unsafe_candidate_id")
        path = (locked / row["image_path"]).resolve()
        if locked not in path.parents or Path(row["image_path"]).is_absolute():
            raise ValueError("unsafe_image_path")
        if sha256(path) != row["sha256"]:
            raise ValueError("locked_image_hash_mismatch:" + row["image_id"])
        with Image.open(path) as im:
            im.load()
            if im.size != (row["width"], row["height"]):
                raise ValueError("image_dimensions_mismatch")
        task = task_by_id[row["image_id"]]
        if task["source_sha256"] != row["sha256"] or task["image_path"] != row["image_path"]:
            raise ValueError("task_identity_mismatch")
        if task["source_graph"] is not None:
            raise ValueError("expected_unfilled_locked_tasks")
    return manifest, {
        "manifest_sha256": sha256(locked / "manifest.json"),
        "tasks_sha256": sha256(locked / "source_graph_tasks.jsonl"),
        "image_count": 30,
        "counts": dict(Counter(r["dataset"] for r in rows)),
    }


def probability(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError("invalid_confidence")
    return float(value)


def validate_box(box, width, height):
    if len(box) != 4 or any(
        isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in box
    ):
        raise ValueError("invalid_bbox")
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("bbox_outside_image_or_degenerate")
    return list(box)


def iou(a, b):
    def area(x):
        return (x[2] - x[0]) * (x[3] - x[1])

    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return inter / (area(a) + area(b) - inter)


def graph_from_raw(raw, row, config):
    """Merge same-label boxes deterministically; retain directed triple evidence.

    Raw hypotheses (including rejected ones) remain in raw/<candidate_id>.json.
    No cap or truncation is applied to meet Task B budgets.
    """
    width, height = row["width"], row["height"]
    detections, nodes, node_meta, mapping = [], [], {}, {}
    seen = set()
    for d in raw["detections"]:
        if d["id"] in seen:
            raise ValueError("duplicate_raw_detection")
        seen.add(d["id"])
        confidence = probability(d["confidence"])
        box = validate_box(d["bbox_xyxy"], width, height)
        if not isinstance(d["label"], str) or not d["label"].strip():
            raise ValueError("invalid_detection_label")
        if confidence >= config["object_threshold"]:
            detections.append({**d, "bbox_xyxy": box})
    for d in sorted(detections, key=lambda d: (-d["confidence"], d["id"])):
        matches = [
            n
            for n in nodes
            if n.label == d["label"] and iou(node_meta[n.id]["bbox_xyxy"], d["bbox_xyxy"]) >= config["merge_iou"]
        ]
        if matches:
            identifier = max(matches, key=lambda n: iou(node_meta[n.id]["bbox_xyxy"], d["bbox_xyxy"])).id
            node_meta[identifier]["raw_detection_ids"].append(d["id"])
        else:
            identifier = f"n{len(nodes) + 1:03d}"
            nodes.append(Node(identifier, d["label"]))
            node_meta[identifier] = {
                "bbox_xyxy": d["bbox_xyxy"],
                "confidence": d["confidence"],
                "confidence_kind": "reltr_softmax_uncalibrated",
                "raw_detection_ids": [d["id"]],
                "attributes": [],
                "origin": "reltr",
            }
        mapping[d["id"]] = identifier
    edges, edge_meta, triples, relation_ids = [], {}, {}, set()
    for r in sorted(raw["relations"], key=lambda r: (-r["confidence"], r["id"])):
        if r["id"] in relation_ids:
            raise ValueError("duplicate_raw_relation")
        relation_ids.add(r["id"])
        probability(r["confidence"])
        if r["subject"] not in seen or r["object"] not in seen:
            raise ValueError("dangling_raw_relation")
        if r["confidence"] < config["relation_threshold"] or r["subject"] not in mapping or r["object"] not in mapping:
            continue
        sub, obj = mapping[r["subject"]], mapping[r["object"]]
        if sub == obj:  # an endpoint merge is not evidence for a self-relation
            continue
        key = (sub, r["predicate"], obj)
        if key in triples:
            edge_meta[triples[key]]["raw_relation_ids"].append(r["id"])
            continue
        identifier = f"e{len(edges) + 1:03d}"
        triples[key] = identifier
        edges.append(Edge(identifier, *key))
        edge_meta[identifier] = {
            "confidence": r["confidence"],
            "confidence_kind": "reltr_predicate_softmax_uncalibrated",
            "triplet_score": r["triplet_score"],
            "raw_relation_ids": [r["id"]],
            "origin": "reltr",
        }
    graph = EditGraph(tuple(nodes), tuple(edges))
    return graph, {"nodes": node_meta, "edges": edge_meta}


def parse_graph(payload):
    # The historical dataclass accepts any iterable for attributes; JSON review
    # must not accidentally turn a string such as "red" into three attributes.
    for node in payload["nodes"]:
        if not isinstance(node.get("attributes", ()), (list, tuple)):
            raise ValueError("node_attributes_must_be_an_array")
    return EditGraph.from_dict(payload)


def validate_record(record):
    if (
        record["candidate_id"] != record["image_id"]
        or record["image_id"] != record["image"]["image_id"]
        or record["source_sha256"] != record["image"]["sha256"]
    ):
        raise ValueError("candidate_image_identity_mismatch")
    graph = parse_graph(record["source_graph"])
    metadata = record["grounding"]
    if set(metadata["nodes"]) != {n.id for n in graph.nodes} or set(metadata["edges"]) != {e.id for e in graph.edges}:
        raise ValueError("grounding_identity_mismatch")
    for n in graph.nodes:
        m = metadata["nodes"][n.id]
        validate_box(m["bbox_xyxy"], record["image"]["width"], record["image"]["height"])
        if m["confidence"] is not None:
            probability(m["confidence"])
        if {a["label"] for a in m["attributes"]} != set(n.attributes) or len(m["attributes"]) != len(n.attributes):
            raise ValueError("attribute_evidence_mismatch")
        for a in m["attributes"]:
            if a["confidence"] is not None:
                probability(a["confidence"])
    for m in metadata["edges"].values():
        if m["confidence"] is not None:
            probability(m["confidence"])
    return graph


def make_record(row, graph, grounding, provenance):
    record = {
        "schema": VERSION,
        "candidate_id": row["image_id"],
        "image_id": row["image_id"],
        "image": row,
        "source_sha256": row["sha256"],
        "source_graph": graph.to_dict(),
        "grounding": grounding,
        "provenance": provenance,
        "annotation_status": "needs_review" if graph.nodes else "empty_prediction_needs_review",
        "human_reviewed": False,
    }
    validate_record(record)
    return record
