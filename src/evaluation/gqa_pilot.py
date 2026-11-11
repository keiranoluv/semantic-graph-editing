"""Explicit GQA taxonomy wrapper around the unchanged reference PredCls matcher."""

from src.evaluation.reference_predcls import evaluate_reference_image, summarize_reference


def evaluate_gqa_image(logits, graph, num_predicates, *, graph_constraint=True):
    if logits.ndim != 2 or logits.shape[1] != num_predicates + 1:
        raise ValueError("GQA logits must match the pinned foreground vocabulary plus background")
    return evaluate_reference_image(
        logits,
        graph.edge_label_index,
        graph.gt_relation_index,
        graph.gt_predicate,
        graph.boxes,
        graph.object_labels,
        graph_constraint=graph_constraint,
    )


def summarize_gqa(results, num_predicates, *, graph_constraint=True):
    summary = summarize_reference(results, num_predicates=num_predicates, graph_constraint=graph_constraint)
    summary["protocol"] = (
        f"GQA_custom_PredCls_{'GC' if graph_constraint else 'NG'}_reference_iou50_"
        f"duplicate_GT_rows_image_macro_per_predicate_{num_predicates}_v1"
    )
    summary["absent_predicates_in_mean"] = "zero; all pinned foreground predicates remain in denominator"
    return summary
