"""GQA baseline model construction and validation metrics."""

import torch
from common.evaluation.reference_predcls import evaluate_reference_image, summarize_reference
from common.models.gat import GATPredicateClassifier
from common.models.mlp_semantic import SemanticMLPPredicateClassifier
from task2_gqa.training.supervision import collate, cross_entropy_sum, forward


def build_model(kind, config):
    kwargs = {
        k: config[k] for k in ("geometry_dim", "num_object_classes", "embedding_dim", "hidden_dim", "num_classes")
    }
    if kind == "gat":
        return GATPredicateClassifier(**kwargs, num_heads=4, num_layers=2, dropout=0.1)
    if kind == "mlp":
        return SemanticMLPPredicateClassifier(**kwargs)
    raise ValueError("Unknown model")


@torch.inference_mode()
def evaluate(model, kind, graphs, config, device, *, include_ng=False):
    model.eval()
    gc, ng = [], []
    loss_sum, count = 0.0, 0
    for begin in range(0, len(graphs), config["batch_size"]):
        group = graphs[begin : begin + config["batch_size"]]
        batch, slices = collate(group, config["num_classes"])
        logits = forward(model, kind, batch, device).cpu()
        loss_sum += float(cross_entropy_sum(logits, batch["targets"]))
        count += len(logits)
        for graph, (start, end) in zip(group, slices, strict=True):
            gc.append(evaluate_gqa_image(logits[start:end], graph, config["num_classes"] - 1))
            if include_ng:
                ng.append(
                    evaluate_gqa_image(logits[start:end], graph, config["num_classes"] - 1, graph_constraint=False)
                )
    if not count:
        raise ValueError("No validation candidate pairs")
    result = {"loss_all_pairs": loss_sum / count, "gc": summarize_gqa(gc, config["num_classes"] - 1)}
    if include_ng:
        result["ng"] = summarize_gqa(ng, config["num_classes"] - 1, graph_constraint=False)
    return result


def compact_metrics(result):
    return {
        "loss_all_pairs": result["loss_all_pairs"],
        **{k: result["gc"][k] for k in ("R@50", "R@100", "mR@50", "mR@100")},
    }


@torch.inference_mode()
def evaluate_fast(model, kind, graphs, config, device, *, include_ng=False):
    from task2_gqa.batching import collate, model_inputs

    model.eval()
    results = {"gc": [], "ng": []}
    total, count = 0.0, 0
    for start in range(0, len(graphs), config["batch_size"]):
        group = graphs[start : start + config["batch_size"]]
        batch = collate(group, config["num_classes"])
        logits = model(**model_inputs(batch, device)).cpu()
        total += float(cross_entropy_sum(logits, batch["targets"]))
        count += len(logits)
        for i, graph in enumerate(group):
            a, b = batch["query_ptr"][i : i + 2].tolist()
            for protocol in ("gc", "ng") if include_ng else ("gc",):
                results[protocol].append(
                    evaluate_gqa_image(
                        logits[a:b],
                        graph,
                        config["num_classes"] - 1,
                        graph_constraint=protocol == "gc",
                    )
                )
    if not count:
        raise ValueError("No validation candidate pairs")
    return {
        "loss_all_pairs": total / count,
        **{
            key: summarize_gqa(rows, config["num_classes"] - 1, graph_constraint=key == "gc")
            for key, rows in results.items()
            if key == "gc" or include_ng
        },
    }


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
