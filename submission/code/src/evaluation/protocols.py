"""Explicit dispatch; historical training keeps its original evaluation protocol."""

from src.evaluation.reference_predcls import evaluate_reference_image
from src.evaluation.scene_graph import evaluate_image


def evaluate_graph_image(logits, graph, protocol="legacy"):
    args = (logits, graph.edge_label_index, graph.gt_relation_index, graph.gt_predicate)
    if protocol == "legacy":
        return evaluate_image(*args)
    if protocol == "reference-v1":
        return evaluate_reference_image(*args, graph.boxes, graph.object_labels)
    if protocol == "reference-ng-v1":
        return evaluate_reference_image(*args, graph.boxes, graph.object_labels, graph_constraint=False)
    raise ValueError(f"Unknown evaluation protocol: {protocol}")
