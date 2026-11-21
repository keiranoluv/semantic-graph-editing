"""Explicit stage denominators, unavailable metrics and policy comparisons."""

from collections import Counter

from task2_graph_editing.he2020.preprocessing.compiler import STAGES, operation_counts


def distribution(values):
    values = sorted(values)
    if not values:
        return None
    return {
        "evaluated": len(values),
        "min": values[0],
        "median": (values[(len(values) - 1) // 2] + values[len(values) // 2]) / 2,
        "mean": sum(values) / len(values),
        "max": values[-1],
        "histogram": dict(sorted(Counter(values).items())),
    }


def summarize(records, alternative):
    n = len(records)
    stages = {}
    for stage in STAGES:
        values = [r["stages"][stage] for r in records]
        stages[stage] = {
            "passed": values.count(True),
            "failed": values.count(False),
            "not_evaluated": values.count(None),
            "cohort": n,
        }
    candidate = sum(r["status"] == "candidate" for r in records)
    parsed = [r for r in records if r["stages"]["graph_parsing"]]
    sized = [r for r in records if "sizes" in r]
    diffed = [r for r in records if "operation_count" in r]
    raw_edges = [r for r in records if r["stages"]["edge_alignment"]]
    return {
        "actual_records_processed": n,
        "synthetic_records": 0,
        "automated_candidates": candidate,
        "automated_acceptance_rate": candidate / n if n else None,
        "stages": stages,
        "edge_alignment": {
            "pairs_passed": len(raw_edges),
            "source_raw_links_evaluated": sum(len(r["raw_source"]["edges"]) for r in raw_edges),
            "target_raw_links_evaluated": sum(len(r["raw_target"]["edges"]) for r in raw_edges),
            "target_canonical_arcs_aligned": sum(len(r["aligned_target_graph"]["edges"]) for r in raw_edges),
        },
        "rejection_reasons": dict(Counter(p for r in records for p in r["problems"])),
        "operation_distribution_all_derived": operation_counts(records),
        "operation_distribution_candidates": operation_counts([r for r in records if r["status"] == "candidate"]),
        "sizes": {key: distribution([r["sizes"][key] for r in sized]) for key in sized[0]["sizes"]} if sized else None,
        "operations_per_example": distribution([r["operation_count"] for r in diffed]),
        "ambiguous_node_identities": sum(
            any(p.startswith("ambiguous_node_identity:") for p in r["problems"]) for r in records
        ),
        "lossy_graph_conversions": sum(r["lossy_conversion"] for r in parsed) if parsed else None,
        "no_graph_changes": sum(r["no_graph_changes"] for r in diffed) if diffed else None,
        "graph_budget_exceeded": sum(any("graph_budget_exceeded" in p for p in r["problems"]) for r in records),
        "operation_budget_exceeded": sum("operation_count_budget_exceeded" in r["problems"] for r in records),
        "action_token_budget_exceeded": sum("action_token_budget_exceeded" in r["problems"] for r in records),
        "raw_id_reordering_detected_pairs": sum(
            any(n["source_raw_id"] != n["target_raw_id"] for n in r.get("alignment", {}).get("node_correspondence", []))
            for r in records
        ),
        "policy_B": {
            "candidates": sum(r["status"] == "candidate" for r in alternative),
            "rejections": dict(Counter(p for r in alternative for p in r["problems"])),
            "operations_on_convertible_subset": operation_counts(alternative),
            "meaning": "attribute-free subset only; no flattening when ownership is unproven",
        },
        "selection": "fixed_seed42_before_conversion_no_failed_sample_replacement",
        "policy": "A_preserve_topology_reciprocal_undirected_v1",
        "direction_recovery": "not_available_in_serialized_Graph; reciprocal_encoding_preserves_undirected_links_only",
        "lossy_conversion_reference": "relative_to_stored_undirected_graphs; paper_scene_relation_directions_are_not_claimed_recovered",
        "historical_node_identity_guarantee": "not_verified; unique_structural_correspondence_only",
        "instruction_semantic_accuracy": None,
        "image_semantic_accuracy": None,
        "human_reviews_completed": 0,
        "human_review_gate_ready": False,
        "at_least_300_requirement_complete": False,
        "assignment_dataset_source_requirement_satisfied": False,
        "instructor_approval": "pending",
        "interpretation": "automated graph/transaction compatibility, not human semantic correctness or original assignment completion",
    }
