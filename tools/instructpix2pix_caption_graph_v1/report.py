"""Stage rates retain the full diagnostic denominator, including failed records."""

from collections import Counter

from src.editing.operations import KINDS


def summarize(labels, *, runtime=None, image_receipts=None):
    n = len(labels)
    candidates = [r for r in labels if r["status"] == "candidate"]
    stages = {}
    for key in (
        "fields",
        "source_parse",
        "target_parse",
        "caption_parsing",
        "alignment",
        "instruction_consistency",
        "structural_validity",
    ):
        passed = sum(r["stages"].get(key) is True for r in labels)
        evaluated = sum(r["stages"].get(key) is not None for r in labels)
        stages[key] = {
            "passed": passed,
            "evaluated": evaluated,
            "diagnostic_denominator": n,
            "rate_all": passed / n if n else None,
            "rate_evaluated": passed / evaluated if evaluated else None,
        }
    operations = Counter(o["kind"] for r in candidates for o in r.get("operations", []))
    images = image_receipts or {}
    return {
        "schema_version": 1,
        "input_records_inspected": n,
        "stages": stages,
        "accepted_candidates": len(candidates),
        "acceptance_rate": len(candidates) / n if n else None,
        "rejected_records": n - len(candidates),
        "rejection_reasons": dict(Counter(p for r in labels for p in r["problems"])),
        "operation_distribution_candidates": {k: operations[k] for k in KINDS},
        "missing_operation_types": [k for k in KINDS if not operations[k]],
        "average_source_nodes_candidates": sum(len(r["source_graph"]["nodes"]) for r in candidates) / len(candidates)
        if candidates
        else None,
        "average_source_edges_candidates": sum(len(r["source_graph"]["edges"]) for r in candidates) / len(candidates)
        if candidates
        else None,
        "average_operations_candidates": sum(len(r["operations"]) for r in candidates) / len(candidates)
        if candidates
        else None,
        "image_availability": dict(Counter(images.get(r["id"], {}).get("status", "not_requested") for r in labels)),
        "runtime": runtime,
        "human_reviews_created": 0,
        "visual_accuracy": None,
        "label_origin": "caption_derived_pseudo_label_candidates_not_visual_ground_truth",
        "dataset_ready": False,
    }
