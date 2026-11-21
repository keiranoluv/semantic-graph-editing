from collections import Counter

from task3_vn100.image_graph.core import validate_record
from task3_vn100.qwen_scoring.scoring import align, dice
from task3_vn100.qwen_scoring.scoring import score as original_score


def score(record, cfg):
    task, grounding = record["task"], record["author"]["grounding"]
    if task["operation"]["kind"] != "attribute":
        result = original_score(record["source"], record["target"], task, grounding, cfg)
    else:
        # A color-only edit should retain all structural nodes/edges. Color E is unobservable.
        source, target = record["source"], record["target"]
        sg, tg = validate_record(source), validate_record(target)
        mapping, alignment, uncertain_s, uncertain_t = align(source, target, cfg)
        reverse = {t: s for s, t in mapping.items()}
        before = {n.id for n in sg.nodes}
        after = {reverse.get(n.id, "target:" + n.id) for n in tg.nodes}
        expected = {(e.subject, e.predicate, e.object) for e in sg.edges}
        observed = {
            (reverse.get(e.subject, "target:" + e.subject), e.predicate, reverse.get(e.object, "target:" + e.object))
            for e in tg.edges
        }
        nd, ed = dice(before, after), dice(expected, observed)
        values = [x["dice"] for x in (nd, ed) if x["dice"] is not None]
        reasons = ["unsupported_color_attribute"]
        if uncertain_s or uncertain_t:
            reasons.append("ambiguous_preservation_alignment")
        if not values:
            reasons.append("no_observable_protected_graph")
        result = {
            "status": "abstained",
            "edit_score": None,
            "preservation_score": sum(values) / len(values) if values else None,
            "graph_consistency": None,
            "reason_codes": reasons,
            "node_preservation": nd,
            "edge_preservation": ed,
            "alignment": alignment,
            "missing_protected_edges": sorted(expected - observed),
            "extra_protected_edges": sorted(observed - expected),
            "warnings": ["structural_preservation_only_color_not_observed"],
            "prompt_coverage": "structural_preservation_only",
            "interpretation": "not_image_accuracy_or_ground_truth_F1",
        }
    # Keep inherited diagnostic P, but do not correlate it as reliable when matching is ambiguous.
    result["preservation_evaluable"] = (
        result["preservation_score"] is not None and "ambiguous_preservation_alignment" not in result["reason_codes"]
    )
    return result


def summarize(rows):
    def group(items):
        return {
            "pairs": len(items),
            "scored": sum(r["scores"]["graph_consistency"] is not None for r in items),
            "edit_available": sum(r["scores"]["edit_score"] is not None for r in items),
            "preservation_evaluable": sum(r["scores"]["preservation_evaluable"] for r in items),
            "reasons": dict(Counter(c for r in items for c in r["scores"]["reason_codes"])),
        }

    summary = group(rows)
    summary.update(
        coverage=summary["scored"] / len(rows) if rows else None,
        by_operation={
            k: group([r for r in rows if r["operation"] == k]) for k in sorted({r["operation"] for r in rows})
        },
        human_labels_used=0,
        spearman=None,
        object_relation_GT_F1=None,
        interpretation="graph evidence coverage, not image pass rate",
    )
    return summary
