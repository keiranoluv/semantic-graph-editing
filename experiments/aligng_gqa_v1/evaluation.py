"""Unchanged GQA evaluator, explicit image pointers from the P0 adapter."""

import torch

from experiments.aligng_adapted_v1.batching import collate, model_inputs
from src.evaluation.gqa_pilot import evaluate_gqa_image, summarize_gqa
from src.training.gqa_pilot import cross_entropy_sum


@torch.inference_mode()
def evaluate(model, kind, graphs, config, device, *, include_ng=False):
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
