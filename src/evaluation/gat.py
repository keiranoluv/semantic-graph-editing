"""GAT evaluation uses the exact same image-level PredCls evaluator as MLPs."""

import torch
import torch.nn.functional as F

from src.datasets.batching import graph_batches
from src.evaluation.scene_graph import evaluate_image, summarize_evaluation
from src.graph.gnn_batching import collate_gnn_graphs, forward_gat_batch


@torch.inference_mode()
def evaluate_gat_batch(model, graphs, device, compute_loss=False):
    """Batched forward, independently rank predictions within each image."""
    if not graphs:
        return [], 0.0, 0
    batch, slices, skipped = collate_gnn_graphs(graphs)
    if skipped:
        raise AssertionError("Evaluation must never skip an image")
    logits = forward_gat_batch(model, batch, device)
    loss_sum = 0.0
    total_pairs = 0
    if compute_loss:
        targets = batch["targets"].to(device)
        mask = targets >= 0
        if mask.any():
            loss_sum = F.cross_entropy(logits[mask], targets[mask], reduction="sum").item()
            total_pairs = int(mask.sum().item())

    logits_cpu = logits.cpu()
    results = []
    for graph, (start, end) in zip(graphs, slices, strict=True):
        results.append(
            evaluate_image(
                logits=logits_cpu[start:end],
                candidate_pairs=graph.edge_label_index,
                gt_pairs=graph.gt_relation_index,
                gt_predicates=graph.gt_predicate,
            )
        )
    return results, loss_sum, total_pairs


@torch.inference_mode()
def validate_gat(model, dataset, limit, device, batch_size=32, batch_loader=None):
    """Full validation: original graph-constrained mR@50 for model selection."""
    from tqdm.auto import tqdm

    model.eval()
    total_loss = 0.0
    total_pairs = 0
    results = []
    for _, graphs in tqdm(
        graph_batches(dataset, limit, batch_size, batch_loader),
        total=(limit + batch_size - 1) // batch_size,
        desc="Validate GAT",
        unit="batch",
    ):
        batch_results, loss_sum, count = evaluate_gat_batch(model, graphs, device, compute_loss=True)
        results.extend(batch_results)
        total_loss += loss_sum
        total_pairs += count
    if total_pairs == 0:
        raise RuntimeError("No validation candidate pairs")
    summary = summarize_evaluation(results)
    summary["loss"] = total_loss / total_pairs
    return summary
