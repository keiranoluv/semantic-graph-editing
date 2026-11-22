"""CPU contracts: immutable pilot inputs, EditGraph sidecars and postprocessing."""

import hashlib
import json
import math
from pathlib import Path

from common.editing.operations import EditGraph

VERSION = "level3-image-graph-v1"


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
