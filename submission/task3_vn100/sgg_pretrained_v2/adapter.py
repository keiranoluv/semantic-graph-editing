"""Strict native array -> EditGraph conversion; no semantic invention."""

import numpy as np
from common.editing.operations import Edge, EditGraph, Node
from task3_vn100.image_graph.core import make_record, validate_box


def integer(x):
    if not np.isfinite(x) or int(x) != x:
        raise ValueError("non_integer_native_index")
    return int(x)


def adapt(row, model, objects, relations, provenance, native):
    nodes, edges = [], []
    grounding = {"nodes": {}, "edges": {}}
    for i, obj in enumerate(objects):
        identifier = f"n{i:03d}"
        box = list(map(float, obj["bbox_xyxy"]))
        validate_box(box, row["width"], row["height"])
        nodes.append(Node(identifier, obj["label"]))
        grounding["nodes"][identifier] = {
            **obj,
            "bbox_xyxy": box,
            "attributes": [],
            "confidence_kind": "detector_score_not_calibrated_on_pilot",
            "raw_detection_ids": [obj.get("native_index", i)],
        }
    seen = {}
    for i, rel in enumerate(relations):
        s, o = integer(rel["subject"]), integer(rel["object"])
        if not 0 <= s < len(nodes) or not 0 <= o < len(nodes):
            raise ValueError("dangling_native_relation")
        key = (s, rel["predicate"], o)
        if key in seen:
            grounding["edges"][seen[key]]["native_rows"].append(i)
            continue
        identifier = f"e{len(edges):03d}"
        edges.append(Edge(identifier, nodes[s].id, rel["predicate"], nodes[o].id))
        grounding["edges"][identifier] = {**rel, "native_rows": [i]}
        seen[key] = identifier
    record = make_record(row, EditGraph(tuple(nodes), tuple(edges)), grounding, provenance)
    record.update(
        model=model,
        status="ok",
        native_output=native,
        attribute_coverage="unsupported_no_attribute_head",
        identity_scope="per_image_only_not_cross_image_correspondence",
    )
    return record


def react_decode(boxes, rels, obj_names, rel_names, ratio, pad, width, height, cfg):
    boxes, rels = np.asarray(boxes), np.asarray(rels)
    if boxes.ndim != 2 or boxes.shape[1] != 6 or rels.ndim != 2 or rels.shape[1] != 5:
        raise ValueError("unexpected_react_output_shape")
    if not np.isfinite(boxes).all() or not np.isfinite(rels).all() or ratio <= 0:
        raise ValueError("nonfinite_native_output")
    for s, o, p, _, _ in rels:
        if not 0 <= integer(s) < len(boxes) or not 0 <= integer(o) < len(boxes):
            raise ValueError("dangling_native_relation")
        if integer(p) not in rel_names:
            raise ValueError("unknown_relation_class")
    for b in boxes:
        if integer(b[4]) not in obj_names:
            raise ValueError("unknown_object_class")
    selected_rels = [r for r in rels if r[3] >= cfg["relation_threshold"]]
    involved = {integer(r[j]) for r in selected_rels for j in (0, 1)}
    selected = [i for i, b in enumerate(boxes) if b[5] >= cfg["box_threshold"] or i in involved]
    mapping = {old: new for new, old in enumerate(selected)}
    objects, adjustments = [], []
    for i in selected:
        raw = (boxes[i, :4] - np.array([*pad, *pad])) / ratio
        box = np.clip(raw, 0, [width, height, width, height]).tolist()
        validate_box(box, width, height)
        if box != raw.tolist():
            adjustments.append({"native_index": i, "before": raw.tolist(), "after": box})
        objects.append(
            {
                "label": obj_names[integer(boxes[i, 4])],
                "confidence": float(boxes[i, 5]),
                "bbox_xyxy": box,
                "native_index": i,
            }
        )
    relations = [
        {
            "subject": mapping[integer(r[0])],
            "object": mapping[integer(r[1])],
            "predicate": rel_names[integer(r[2])],
            "confidence": float(r[4]),
            "triplet_confidence": float(r[3]),
            "confidence_kind": "relation_score",
            "native_index": i,
        }
        for i, r in enumerate(rels)
        if r[3] >= cfg["relation_threshold"]
    ]
    return objects, relations, {"selected_native_indices": selected, "clipped_boxes": adjustments}
