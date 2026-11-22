"""Graph evidence only: no image-generation calls, outcome labels, or fitted weights."""

import numpy as np
from scipy.optimize import linear_sum_assignment
from task3_vn100.image_graph.core import validate_record
from task3_vn100.shared.geometry import inside, iou, label


def boxes(record):
    w, h = record["image"]["width"], record["image"]["height"]
    return {
        key: [v / (w if i % 2 == 0 else h) for i, v in enumerate(m["bbox_xyxy"])]
        for key, m in record["grounding"]["nodes"].items()
    }


def align(source, target, cfg):
    """Maximum-cardinality compatible assignment, then maximize normalized box IoU."""
    sg, tg = validate_record(source), validate_record(target)
    sb, tb = boxes(source), boxes(target)
    n, m = len(sg.nodes), len(tg.nodes)
    if not n or not m:
        return {}, [], set(), set()
    overlap = np.zeros((n, m))
    weights = np.zeros((n, m + n))  # dummies permit unmatched source instances
    for i, a in enumerate(sg.nodes):
        for j, b in enumerate(tg.nodes):
            if label(a.label) == label(b.label):
                value = iou(sb[a.id], tb[b.id])
                if value >= cfg["matching_iou"]:
                    overlap[i, j] = value
                    weights[i, j] = n + m + 1 + value
    ri, ci = linear_sum_assignment(weights, maximize=True)
    mapping, evidence, uncertain_s, uncertain_t = {}, [], set(), set()
    for i, j in zip(ri, ci, strict=True):
        if j >= m or weights[i, j] == 0:
            continue
        sid, tid = sg.nodes[i].id, tg.nodes[j].id
        alternatives = [overlap[i, k] for k in range(m) if k != j and overlap[i, k] > 0]
        alternatives += [overlap[k, j] for k in range(n) if k != i and overlap[k, j] > 0]
        ambiguous = bool(alternatives and overlap[i, j] - max(alternatives) < cfg["ambiguity_margin"])
        mapping[sid] = tid
        evidence.append({"source": sid, "target": tid, "iou": float(overlap[i, j]), "ambiguous": ambiguous})
        if ambiguous:
            uncertain_s.add(sid)
            uncertain_t.add(tid)
    # A rejected detection competing for an already assigned instance is not proven absent/new.
    for i, a in enumerate(sg.nodes):
        if a.id not in mapping and overlap[i].max() > 0:
            uncertain_s.add(a.id)
    for j, b in enumerate(tg.nodes):
        if b.id not in mapping.values() and overlap[:, j].max() > 0:
            uncertain_t.add(b.id)
    return mapping, evidence, uncertain_s, uncertain_t


def dice(a, b):
    common = a & b
    return {
        "common": len(common),
        "source_count": len(a),
        "target_count": len(b),
        "dice": 2 * len(common) / (len(a) + len(b)) if a or b else None,
    }


def score(source, target, task, grounding, cfg):
    sg, tg = validate_record(source), validate_record(target)
    sm, tm = {n.id: n for n in sg.nodes}, {n.id: n for n in tg.nodes}
    sb, tb = boxes(source), boxes(target)
    mapping, alignment, uncertain_s, uncertain_t = align(source, target, cfg)
    reverse = {t: s for s, t in mapping.items()}
    op = task["operation"]
    kind, sid, region = op["kind"], grounding["source_node"], grounding["region"]
    if kind not in ("add", "remove", "replace"):
        raise ValueError("unsupported_operation")
    if sid is not None and (sid not in sm or label(sm[sid].label) != label(op["label"])):
        raise ValueError("invalid_source_anchor")
    minimum = cfg["min_edit_object_confidence"]

    def confidence(rec, key):
        return rec["grounding"]["nodes"][key]["confidence"]

    checks, reasons, warnings, allowed = [], [], [], set()
    affected = {sid} if sid is not None and kind in ("remove", "replace") else set()

    def count(nodes, name):
        return sum(label(n.label) == label(name) for n in nodes.values())

    def check(name, value, reason=None, **details):
        checks.append({"criterion": name, "value": value, "reason": reason, **details})
        if reason and value is None:
            reasons.append(reason)

    binding_problem = None
    if grounding["binding_status"] == "ambiguous_instruction":
        binding_problem = "ambiguous_source_instruction"
    elif kind in ("remove", "replace"):
        if sid is None:
            binding_problem = "requested_source_object_not_in_graph"
        elif confidence(source, sid) is None or confidence(source, sid) < minimum:
            binding_problem = "source_anchor_low_confidence"
    if kind in ("remove", "replace"):
        before, after = count(sm, op["label"]), count(tm, op["label"])
        if binding_problem:
            check("requested_instance_removed", None, binding_problem)
        elif sid in uncertain_s:
            check("requested_instance_removed", None, "ambiguous_edit_anchor_alignment")
        elif sid in mapping:
            tid = mapping[sid]
            if confidence(target, tid) is None or confidence(target, tid) < minimum:
                check("requested_instance_removed", None, "target_anchor_low_confidence", target=tid)
            else:
                check("requested_instance_removed", 0.0, "source_instance_still_matched", target=tid)
        else:
            relabel = [
                n.id
                for n in tg.nodes
                if n.id not in reverse
                and label(n.label) != label(op["label"])
                and not (kind == "replace" and label(n.label) == label(op["replacement_label"]))
                and iou(sb[sid], tb[n.id]) >= cfg["relabel_iou"]
            ]
            if relabel:
                check("requested_instance_removed", None, "possible_detector_relabel", candidates=relabel)
            elif after >= before:
                check(
                    "requested_instance_removed",
                    None,
                    "possible_movement_or_count_instability",
                    category_counts=[before, after],
                )
            else:
                check("requested_instance_removed", 1.0, category_counts=[before, after])
                warnings.append("detector_absence_is_not_visual_proof_of_removal")
    if kind in ("add", "replace"):
        wanted = op["replacement_label"] if kind == "replace" else op["label"]
        candidates = [
            n.id
            for n in tg.nodes
            if label(n.label) == label(wanted) and n.id not in reverse and inside(tb[n.id], region)
        ]
        strong = [t for t in candidates if confidence(target, t) is not None and confidence(target, t) >= minimum]
        strong.sort(key=lambda t: (-confidence(target, t), t))
        before, after = count(sm, wanted), count(tm, wanted)
        if binding_problem:
            check("new_instance_in_requested_region", None, binding_problem)
        elif any(t in uncertain_t for t in strong):
            check("new_instance_in_requested_region", None, "ambiguous_new_instance_alignment", candidates=strong)
        elif strong:
            selected = strong[0]
            missing_same = [s for s, n in sm.items() if label(n.label) == label(wanted) and s not in mapping]
            relabel = [
                s
                for s, n in sm.items()
                if s not in mapping
                and s not in affected
                and label(n.label) != label(wanted)
                and iou(sb[s], tb[selected]) >= cfg["relabel_iou"]
            ]
            if after <= before or missing_same:
                check(
                    "new_instance_in_requested_region",
                    None,
                    "possible_movement_or_count_instability",
                    candidates=strong,
                    category_counts=[before, after],
                )
            elif relabel:
                check("new_instance_in_requested_region", None, "possible_detector_relabel", candidates=relabel)
            else:
                allowed.add(selected)  # only one requested instance is exempted from preservation
                check(
                    "new_instance_in_requested_region",
                    1.0,
                    selected=selected,
                    candidates=strong,
                    category_counts=[before, after],
                )
        elif candidates:
            check("new_instance_in_requested_region", None, "new_instance_low_confidence", candidates=candidates)
        else:
            check("new_instance_in_requested_region", 0.0, "no_new_requested_instance_in_region")
            warnings.append("missing_target_detection_can_hide_a_successful_visual_edit")
    protected = set(sm) - affected
    excluded_target = {mapping[s] for s in affected if s in mapping} | allowed
    eligible_target = set(tm) - excluded_target
    # Express matched target IDs in source coordinates; unmatched IDs stay explicitly target-local.
    observed_nodes = {reverse.get(t, "target:" + t) for t in eligible_target}
    node_dice = dice(protected, observed_nodes)
    expected_edges = {
        (e.subject, e.predicate, e.object) for e in sg.edges if e.subject in protected and e.object in protected
    }
    observed_edges = {
        (reverse.get(e.subject, "target:" + e.subject), e.predicate, reverse.get(e.object, "target:" + e.object))
        for e in tg.edges
        if e.subject in eligible_target and e.object in eligible_target
    }
    edge_dice = dice(expected_edges, observed_edges)
    parts = [x["dice"] for x in (node_dice, edge_dice) if x["dice"] is not None]
    preservation = sum(parts) / len(parts) if parts else None
    edit = (
        sum(c["value"] for c in checks) / len(checks)
        if checks and all(c["value"] is not None for c in checks)
        else None
    )
    diagnostic = edit * preservation if edit is not None and preservation is not None else None
    uncertain_protected = sorted((uncertain_s & protected) | {"target:" + t for t in uncertain_t & eligible_target})
    if uncertain_protected:
        reasons.append("ambiguous_preservation_alignment")
    if preservation is None:
        reasons.append("no_observable_protected_graph")
    if edge_dice["dice"] is None:
        warnings.append("no_protected_relations_to_evaluate")
    if len(protected) < 3:
        warnings.append("limited_protected_node_evidence")
    overall = diagnostic if not reasons else None
    return {
        "status": "scored" if overall is not None else "abstained",
        "edit_score": edit,
        "preservation_score": preservation,
        "graph_consistency": overall,
        "diagnostic_product_before_abstention": diagnostic,
        "reason_codes": sorted(set(reasons)),
        "warnings": sorted(set(warnings)),
        "checks": checks,
        "alignment": alignment,
        "ambiguous_protected_nodes": uncertain_protected,
        "affected_source_nodes": sorted(affected),
        "allowed_new_target_nodes": sorted(allowed),
        "protected_source_nodes": sorted(protected),
        "eligible_target_nodes": sorted(eligible_target),
        "unmatched_source_nodes": sorted(set(sm) - set(mapping)),
        "unmatched_target_nodes": sorted(set(tm) - set(reverse)),
        "node_preservation": node_dice,
        "edge_preservation": edge_dice,
        "missing_protected_edges": sorted(expected_edges - observed_edges),
        "extra_protected_edges": sorted(observed_edges - expected_edges),
        "prompt_coverage": "category_instance_and_approximate_2D_region_only",
        "unsupported_constraints": [
            "color",
            "material",
            "fine_grained_subtype",
            "pose",
            "size",
            "pixel_texture",
            "depth",
        ],
        "interpretation": "predicted_graph_consistency_not_image_quality_probability_or_ground_truth_F1",
    }
