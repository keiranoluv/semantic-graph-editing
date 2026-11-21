"""GQA baseline model construction and validation metrics."""

import torch
from common.evaluation.gqa_pilot import evaluate_gqa_image, summarize_gqa
from common.models.gat import GATPredicateClassifier
from common.models.mlp_semantic import SemanticMLPPredicateClassifier
from common.training.gqa_pilot import collate, cross_entropy_sum, forward


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
