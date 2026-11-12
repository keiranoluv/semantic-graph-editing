"""Strict local edits. Instructions select checks, never count as visual evidence."""

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.editing.operations import Edge, EditGraph, Node, apply_operations, diff_aligned_graphs
from tools.magicbrush_graphs import iou
from tools.magicbrush_grounded_v1.language import scope


def cosine(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(a @ b / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-12))


def color(node, config):
    scores = sorted(node.get("color_scores", {}).items(), key=lambda x: (-x[1], x[0]))
    if len(scores) < 2 or scores[0][1] - scores[1][1] < config["color_margin"]:
        return None
    winner = scores[0][0]
    return winner if node.get("color_fractions", {}).get(winner, 0) >= config["color_min_pixel_fraction"] else None


def spatial_edges(nodes, boxes, margin):
    """Deterministic center-order edges, not physical/support/action relations.

    Ordered IDs keep edge identity stable when the predicate flips. Near ties
    have no edge. Pixel boxes are sidecar evidence, not graph node attributes.
    """
    edges = []
    ids = sorted(n.id for n in nodes)
    for index, a in enumerate(ids):
        for b in ids[index + 1 :]:
            for axis, (negative, positive) in enumerate((("left_of", "right_of"), ("above", "below"))):
                delta = (boxes[a][axis] + boxes[a][axis + 2] - boxes[b][axis] - boxes[b][axis + 2]) / 2000
                if abs(delta) >= margin:
                    edges.append(Edge(f"spatial-{a}-{b}-{axis}", a, negative if delta < 0 else positive, b))
    return tuple(edges)


def correspondence(left, right, config, moving_label=None):
    n, m = len(left), len(right)
    costs = np.full((n, m), 5.0)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            similarity, overlap = cosine(a["embedding"], b["embedding"]), iou(a["bbox"], b["bbox"])
            if (
                a["label"] == b["label"]
                and similarity >= config["match_min_cosine"]
                and (overlap >= config["match_min_iou"] or a["label"] == moving_label and similarity >= 0.93)
            ):
                costs[i, j] = 0.55 * (1 - similarity) + 0.45 * (1 - overlap)
    matrix = np.full((n + m, n + m), 5.0)
    matrix[:n, :m] = costs
    for i in range(n):
        matrix[i, m + i] = 0.4
    for j in range(m):
        matrix[n + j, j] = 0.4
    matrix[n:, m:] = 0
    matches, ambiguous = {}, []
    if n + m:
        for i, j in zip(*linear_sum_assignment(matrix), strict=True):
            if i >= n or j >= m or costs[i, j] >= 5:
                continue
            alternatives = [costs[i, k] for k in range(m) if k != j] + [costs[k, j] for k in range(n) if k != i]
            if min(alternatives, default=5.0) - costs[i, j] < config["match_margin"]:
                ambiguous.append([i, j])
            else:
                matches[i] = j
    return matches, ambiguous


def local_change(proof, config):
    return proof["cosine"] <= config["absence_max_cosine"] and proof["mae"] >= config["change_min_mae"]


def absent(proposal, label, config):
    # All pre-threshold queries contribute; non-detection alone never passes an edit.
    return proposal["presence_scores"].get(label, 1.0) <= config["absence_max_score"]


def analyze(source, target, evidence, instruction, config):
    intent = scope(instruction)
    base = {
        "status": "unsupported",
        "scope": intent,
        "eligible_for_training": False,
        "human_review_status": "pending",
        "automatic_approval": False,
        "label_origin": "BLIP_DINO_CLIP_source_anchored_spatial_pseudo_labels_not_GT",
        "instruction_used_as_visual_evidence": False,
    }
    if not intent["supported"]:
        return {**base, "problems": [intent["reason"]]}
    if not evidence.get("comparable_aspect_ratio"):
        return {**base, "problems": ["aspect_ratio_change_not_supported"]}
    left, right = source["nodes"], target["nodes"]
    if not left:
        return {**base, "problems": ["no_grounded_source_objects"]}
    before = intent.get("before", {})
    after = intent.get("after", {})
    kind = intent["kind"]
    moving_label = before.get("label") if kind == "move" else None
    matches, ambiguous = correspondence(left, right, config, moving_label)
    problems = ["ambiguous_instance_correspondence"] if ambiguous else []
    if source.get("overflow") or target.get("overflow"):
        problems.append("object_budget_exceeded")
    if source.get("ambiguous_focus") or target.get("ambiguous_focus"):
        problems.append("overlapping_focus_query_labels")
    source_nodes = [
        Node(f"n{i+1:03d}", n["label"], (c,) if (c := color(n, config)) else ()) for i, n in enumerate(left)
    ]
    source_boxes = {n.id: left[i]["bbox"] for i, n in enumerate(source_nodes)}
    graph = EditGraph(tuple(source_nodes), spatial_edges(source_nodes, source_boxes, config["spatial_center_margin"]))
    nodes, boxes = {n.id: n for n in graph.nodes}, dict(source_boxes)
    focus_indices = [i for i, n in enumerate(left) if n["label"] == before.get("label")]
    target_indices = [j for j, n in enumerate(right) if n["label"] == after.get("label")]
    if kind != "add" and len(focus_indices) != 1:
        problems.append("source_focus_reference_not_unique")
    if kind != "delete" and len(target_indices) != 1:
        problems.append("target_focus_reference_not_unique")
    context_indices = [i for i, n in enumerate(left) if n["label"] == intent.get("context", {}).get("label")]
    if kind == "move" and len(context_indices) != 1:
        problems.append("spatial_context_reference_not_unique")
    if before.get("color") and len(focus_indices) == 1 and color(left[focus_indices[0]], config) != before["color"]:
        problems.append("source_requested_color_not_visually_supported")
    changed, focus_boxes, consumed_targets = set(), [], set()
    for i, node in enumerate(left):
        identifier = source_nodes[i].id
        proof = evidence["source_anchors"][node["id"]]
        matched = right[matches[i]] if i in matches else None
        is_focus = len(focus_indices) == 1 and i == focus_indices[0]
        if is_focus and kind == "delete":
            contrast = proof["source_class_score"] - proof["target_class_score"]
            if (
                matched is None
                and absent(target, node["label"], config)
                and local_change(proof, config)
                and contrast >= config["class_contrast_margin"]
            ):
                del nodes[identifier]
                del boxes[identifier]
                changed.add(identifier)
                focus_boxes.append(node["bbox"])
            else:
                problems.append("deletion_has_insufficient_visual_evidence")
        elif is_focus and kind == "replace" and len(target_indices) == 1:
            replacement = right[target_indices[0]]
            reverse_proof = evidence["target_anchors"][replacement["id"]]
            contrast = proof["source_class_score"] - proof["target_class_score"]
            new_contrast = reverse_proof["target_class_score"] - reverse_proof["source_class_score"]
            c = color(replacement, config)
            if after.get("color") and c != after["color"]:
                problems.append("requested_replacement_color_not_verified")
            if (
                not absent(target, node["label"], config)
                or not absent(source, replacement["label"], config)
                or iou(node["bbox"], replacement["bbox"]) < 0.5
                or not local_change(proof, config)
                or min(contrast, new_contrast) < config["class_contrast_margin"]
            ):
                problems.append("replacement_has_insufficient_visual_evidence")
            else:
                nodes[identifier] = Node(identifier, replacement["label"], (c,) if c else ())
                boxes[identifier] = replacement["bbox"]
                changed.add(identifier)
                consumed_targets.add(target_indices[0])
                focus_boxes.extend((node["bbox"], replacement["bbox"]))
        elif is_focus and kind == "color":
            old_color = color(node, config)
            new_color = color(matched, config) if matched else None
            if (
                matched is None
                or old_color is None
                or new_color != after["color"]
                or new_color == old_color
                or iou(node["bbox"], matched["bbox"]) < 0.85
                or proof["mae"] < config["change_min_mae"]
            ):
                problems.append("color_change_has_insufficient_visual_evidence")
            else:
                nodes[identifier] = Node(identifier, node["label"], (new_color,))
                changed.add(identifier)
                focus_boxes.extend((node["bbox"], matched["bbox"]))
                # Attribute edit does not turn detector box jitter into edge changes.
        elif is_focus and kind == "move":
            if matched is None or iou(node["bbox"], matched["bbox"]) >= 0.85 or proof["mae"] < config["change_min_mae"]:
                problems.append("movement_has_insufficient_visual_evidence")
            else:
                boxes[identifier] = matched["bbox"]
                changed.add(identifier)
                focus_boxes.extend((node["bbox"], matched["bbox"]))
        else:
            # Keep exact source identity, attributes and geometry only after checking
            # the same image coordinates. A detector match alone is insufficient.
            if proof["cosine"] < config["unchanged_cosine"] or proof["mae"] > config["unchanged_mae"]:
                problems.append("source_context_not_visually_stable")
    if kind == "add" and len(target_indices) == 1:
        j = target_indices[0]
        added = right[j]
        proof = evidence["target_anchors"][added["id"]]
        contrast = proof["target_class_score"] - proof["source_class_score"]
        c = color(added, config)
        if after.get("color") and c != after["color"]:
            problems.append("requested_addition_color_not_verified")
        if (
            j in matches.values()
            or not absent(source, added["label"], config)
            or not local_change(proof, config)
            or contrast < config["class_contrast_margin"]
        ):
            problems.append("addition_has_insufficient_visual_evidence")
        else:
            nodes["new001"] = Node("new001", added["label"], (c,) if c else ())
            boxes["new001"] = added["bbox"]
            changed.add("new001")
            consumed_targets.add(j)
            focus_boxes.append(added["bbox"])
    # Newly detected context may have been hallucinated or missed in source:
    # abstain rather than silently dropping apparent secondary changes.
    if any(j not in set(matches.values()) | consumed_targets for j in range(len(right))):
        problems.append("unexplained_target_detection")
    target_graph = EditGraph(
        tuple(nodes.values()), spatial_edges(nodes.values(), boxes, config["spatial_center_margin"])
    )
    operations = diff_aligned_graphs(graph, target_graph)
    if kind == "move" and len(focus_indices) == len(context_indices) == 1:
        a, b = source_nodes[focus_indices[0]].id, source_nodes[context_indices[0]].id
        reverse = {"left_of": "right_of", "right_of": "left_of", "above": "below", "below": "above"}
        requested = any(
            (e.subject, e.object, e.predicate) == (a, b, intent["direction"])
            or (e.subject, e.object, reverse[e.predicate]) == (b, a, intent["direction"])
            for e in target_graph.edges
        )
        if not requested or not any(op.kind.endswith("_edge") for op in operations):
            problems.append("requested_spatial_change_not_verified")
    if not changed or not operations:
        problems.append("requested_change_not_verified")
    if len(operations) > config["max_operations"]:
        problems.append("edit_exceeds_operation_budget")
    outside = None
    if focus_boxes and evidence["changed_pixels"]:
        mask = np.zeros_like(evidence["change_mask"], dtype=bool)
        h, w = mask.shape
        for a, b, c, d in focus_boxes:
            mask[
                max(0, int(b * h / 1000)) : min(h, int(np.ceil(d * h / 1000))),
                max(0, int(a * w / 1000)) : min(w, int(np.ceil(c * w / 1000))),
            ] = True
        outside = int((evidence["change_mask"] & ~mask).sum()) / evidence["changed_pixels"]
        if outside > config["max_outside_change_fraction"]:
            problems.append("image_changes_outside_verified_focus")
    elif not evidence["changed_pixels"]:
        problems.append("no_visible_pixel_change")
    if apply_operations(graph, operations) != target_graph:
        raise AssertionError("Candidate operations do not reconstruct target")
    return {
        **base,
        "status": "candidate" if not problems else "needs_review",
        "problems": sorted(set(problems)),
        "passed_automatic_filters": not problems,
        "source_graph": graph.to_dict(),
        "target_graph": target_graph.to_dict(),
        "operations": [op.to_dict() for op in operations],
        "source_boxes": source_boxes,
        "target_boxes": boxes,
        "matches": {left[i]["id"]: right[j]["id"] for i, j in matches.items()},
        "outside_change_fraction": outside,
        "target_reconstruction_exact": True,
        "graph_scope": "partial grounded objects; center-order spatial edges only; no support/action relations",
        "context_policy": "exact source ID/attributes/geometry retained only after image stability checks",
        "box_origin": "source DINO boxes; target uses source anchors for persistent context/color edits and new detections for add/replace/move; not GT",
    }
