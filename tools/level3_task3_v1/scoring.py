"""Geometry-grounded one-to-one correspondence and explicit edit/preservation scores.

These scores measure frozen detector graphs, not correctness of the pixels.
No thresholds are fitted using human outcome labels.
"""

import copy
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from tools.level3_image_graph_v1.core import validate_record

from .core import norm_box


def iou(a, b):
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def label(x):
    return " ".join(x.casefold().split())


def align(source, target, cfg):
    sg, tg = validate_record(source), validate_record(target)
    n, m = len(sg.nodes), len(tg.nodes)
    if not n or not m:
        return {}, []
    weights = np.zeros((n, m + n), dtype=np.float64)  # own dummy targets permit abstention
    overlap = np.zeros((n, m))
    for i, a in enumerate(sg.nodes):
        for j, b in enumerate(tg.nodes):
            if label(a.label) == label(b.label):
                overlap[i, j] = iou(norm_box(source, a.id), norm_box(target, b.id))
                if overlap[i, j] >= cfg["matching_iou"]:
                    weights[i, j] = 1 + overlap[i, j]
    ri, ci = linear_sum_assignment(weights, maximize=True)
    matches = {}
    evidence = []
    for i, j in zip(ri, ci, strict=False):
        if j >= m or weights[i, j] <= 0:
            continue
        sid, tid = sg.nodes[i].id, tg.nodes[j].id
        alternatives = [overlap[i, k] for k in range(m) if k != j] + [overlap[k, j] for k in range(n) if k != i]
        ambiguous = bool(alternatives and overlap[i, j] - max(alternatives) < cfg["ambiguity_margin"])
        matches[sid] = tid
        evidence.append({"source": sid, "target": tid, "iou": float(overlap[i, j]), "ambiguous": ambiguous})
    return matches, evidence


def canonical_source(original, resized, cfg):
    """Keep original anchor IDs only for matched resized-source detections."""
    mapping, evidence = align(original, resized, cfg)
    reverse = {t: s for s, t in mapping.items()}
    out = copy.deepcopy(resized)
    renames = {n["id"]: reverse.get(n["id"], "resize-" + n["id"]) for n in out["source_graph"]["nodes"]}
    for node in out["source_graph"]["nodes"]:
        node["id"] = renames[node["id"]]
    out["grounding"]["nodes"] = {renames[k]: v for k, v in out["grounding"]["nodes"].items()}
    for edge in out["source_graph"]["edges"]:
        edge["subject"] = renames[edge["subject"]]
        edge["object"] = renames[edge["object"]]
    validate_record(out)
    return out, {
        "matches": evidence,
        "missing_original_anchors": sorted(set(n["id"] for n in original["source_graph"]["nodes"]) - set(mapping)),
        "policy": "original_IDs_transferred_by_label_and_normalized_IoU; unmatched_resize_nodes_get_new_IDs",
    }


def inside(box, region):
    x = (box[0] + box[2]) / 2
    y = (box[1] + box[3]) / 2
    return region[0] <= x <= region[2] and region[1] <= y <= region[3]


def f1(tp, fp, fn):
    den = 2 * tp + fp + fn
    return {"tp": tp, "fp": fp, "fn": fn, "f1": 2 * tp / den if den else None}


def score(source, target, task, cfg):
    sg, tg = validate_record(source), validate_record(target)
    sm = {n.id: n for n in sg.nodes}
    tm = {n.id: n for n in tg.nodes}
    mapping, evidence = align(source, target, cfg)
    reverse = {t: s for s, t in mapping.items()}
    op = task["operation"]
    kind = op["kind"]
    affected = set()
    allowed_new = set()
    checks = []
    reason = None
    st = {(e.subject, e.predicate, e.object) for e in sg.edges}
    tt = {(e.subject, e.predicate, e.object) for e in tg.edges}
    if kind == "attribute":
        reason = "unsupported_attribute_head"
    if kind in ("remove", "replace", "attribute"):
        affected = {op["source_node"]}
        if op["source_node"] not in sm:
            reason = "source_anchor_not_observable_after_resize"
    if kind == "relation" and any(op[k] not in sm for k in ("subject", "object")):
        reason = "source_relation_anchor_not_observable_after_resize"
    if reason is None:
        if kind in ("remove", "replace"):
            old_label = label(sm[op["source_node"]].label)
            before_count = sum(label(n.label) == old_label for n in sg.nodes)
            after_count = sum(label(n.label) == old_label for n in tg.nodes)
            checks.append(
                {
                    "criterion": "source_instance_absent",
                    "met": op["source_node"] not in mapping and after_count < before_count,
                    "category_counts": [before_count, after_count],
                    "warning": "detector absence is not visual proof of removal",
                }
            )
        if kind in ("add", "replace"):
            region = op["region"] if kind == "add" else norm_box(source, op["source_node"])
            candidates = [
                n.id
                for n in tg.nodes
                if label(n.label) == label(op["label"])
                and n.id not in reverse
                and inside(norm_box(target, n.id), region)
            ]
            # Credit one new instance, leaving extra detections to preservation precision.
            candidates.sort(key=lambda tid: (-target["grounding"]["nodes"][tid].get("confidence", 0), tid))
            before_count = sum(label(n.label) == label(op["label"]) for n in sg.nodes)
            after_count = sum(label(n.label) == label(op["label"]) for n in tg.nodes)
            is_new = bool(candidates) and after_count > before_count
            if is_new:
                allowed_new.add(candidates[0])
            checks.append(
                {
                    "criterion": "new_instance_in_edit_region",
                    "met": is_new,
                    "candidates": candidates,
                    "category_counts": [before_count, after_count],
                }
            )
        if kind == "relation":
            triple = (mapping.get(op["subject"]), op["predicate"], mapping.get(op["object"]))
            anchors = triple[0] is not None and triple[2] is not None
            checks.append(
                {"criterion": "directed_relation_state", "met": anchors and ((triple in tt) == op["present"])}
            )
    protected = set(sm) - affected
    target_excluded = {mapping[s] for s in affected if s in mapping} | allowed_new
    eligible_target = set(tm) - target_excluded
    matched = {s for s in protected if s in mapping and mapping[s] in eligible_target}
    node_pres = f1(len(matched), len(eligible_target) - len(matched), len(protected) - len(matched))
    expected_edges = {e for e in st if e[0] in protected and e[2] in protected}
    observed_edges = {
        (reverse.get(s, "target:" + s), p, reverse.get(o, "target:" + o))
        for s, p, o in tt
        if s in eligible_target and o in eligible_target
    }
    if kind == "relation":
        changed = (op["subject"], op["predicate"], op["object"])
        expected_edges.discard(changed)
        observed_edges.discard(changed)
    edge_pres = f1(
        len(expected_edges & observed_edges), len(observed_edges - expected_edges), len(expected_edges - observed_edges)
    )
    parts = [x["f1"] for x in (node_pres, edge_pres) if x["f1"] is not None]
    preservation = sum(parts) / len(parts) if parts else None
    edit = sum(c["met"] for c in checks) / len(checks) if checks and reason is None else None
    overall = (
        cfg["edit_weight"] * edit + cfg["preservation_weight"] * preservation
        if edit is not None and preservation is not None
        else None
    )
    return {
        "status": "scored" if overall is not None else "unscorable",
        "reason": reason,
        "edit_score": edit,
        "preservation_score": preservation,
        "graph_consistency": overall,
        "node_preservation": node_pres,
        "edge_preservation": edge_pres,
        "edit_checks": checks,
        "alignment": evidence,
        "ambiguous_matches": sum(x["ambiguous"] for x in evidence),
        "protected_nodes": sorted(protected),
        "affected_nodes": sorted(affected),
        "allowed_new_targets": sorted(allowed_new),
        "unmatched_source": sorted(set(sm) - set(mapping)),
        "unmatched_target": sorted(set(tm) - set(reverse)),
        "interpretation": "heuristic_on_predicted_graphs; not probability or human ground truth",
    }


def correlate(rows, labels):
    """Descriptive Spearman on the SAME complete-case subset for each comparison."""
    from scipy.stats import spearmanr

    keys = ("graph_consistency", "clipscore", "negative_lpips")
    index = {r["case_id"]: r for r in rows}
    if len(index) != len(rows):
        raise ValueError("duplicate_result_case")
    seen = set()
    usable = []
    for review in labels:
        cid = review["case_id"]
        if cid in seen or cid not in index:
            raise ValueError("unknown_or_duplicate_review_case")
        seen.add(cid)
        if review["status"] == "pending":
            continue
        if review["status"] != "reviewed" or not review["reviewer"].strip() or not review["notes"].strip():
            raise ValueError("invalid_outcome_review")
        for field in ("edit_pass", "preservation_pass", "overall_pass"):
            if type(review[field]) is not bool:
                raise ValueError("human_binary_label_required")
        if review["overall_pass"] != (review["edit_pass"] and review["preservation_pass"]):
            raise ValueError("overall_requires_both_axes")
        row = index[cid]
        if any(row.get(k) is None or not math.isfinite(row[k]) for k in keys):
            continue
        usable.append({**row, "label": int(review["overall_pass"])})
    result = {
        "reviewed": sum(r["status"] == "reviewed" for r in labels),
        "common_subset_n": len(usable),
        "common_subset_cases": [r["case_id"] for r in usable],
        "spearman": {},
        "disagreement_candidates": [],
        "interpretation": "development_pilot_descriptive_only; repeated edits of an image are not independent",
    }
    for key in keys:
        if len(usable) < 3 or len({r["label"] for r in usable}) < 2 or len({r[key] for r in usable}) < 2:
            result["spearman"][key] = {"rho": None, "reason": "insufficient_or_constant_data"}
        else:
            result["spearman"][key] = {
                "rho": float(spearmanr([r[key] for r in usable], [r["label"] for r in usable]).statistic),
                "reason": None,
            }
    if usable:
        from scipy.stats import rankdata

        n = len(usable)
        g = rankdata([r["graph_consistency"] for r in usable]) / n
        c = rankdata([r["clipscore"] for r in usable]) / n
        lpips_ranks = rankdata([r["negative_lpips"] for r in usable]) / n
        for i, r in enumerate(usable):
            gap = float(max(abs(g[i] - c[i]), abs(g[i] - lpips_ranks[i])))
            if gap > 0:
                result["disagreement_candidates"].append(
                    {
                        "case_id": r["case_id"],
                        "rank_gap": gap,
                        "analysis": "",
                        "analysis_status": "needs_human_analysis",
                    }
                )
        result["disagreement_candidates"] = sorted(
            result["disagreement_candidates"], key=lambda r: (-r["rank_gap"], r["case_id"])
        )[:10]
    result["ten_disagreements_analyzed"] = False
    return result
