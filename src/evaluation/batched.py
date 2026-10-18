"""Mini-batched MLP inference preserving per-image scene graphs."""

import torch

from src.evaluation.scene_graph import evaluate_image


def collate_evaluation_graphs(graphs):
    """Combine graph inputs and preserve individual image boundaries."""
    xs = []
    labels = []
    pairs = []
    edge_slices = []
    node_offset = 0
    edge_offset = 0

    for graph in graphs:
        count = graph.edge_label_index.shape[1]

        xs.append(graph.x)
        labels.append(graph.object_labels)
        pairs.append(graph.edge_label_index + node_offset)

        edge_slices.append((edge_offset, edge_offset + count))

        node_offset += graph.num_nodes
        edge_offset += count

    batch = {
        "x": torch.cat(xs, dim=0),
        "object_labels": torch.cat(labels, dim=0),
        "pairs": torch.cat(pairs, dim=1),
    }
    return batch, edge_slices


def forward_evaluation_batch(model, batch, name, device):
    """One model forward for all candidate pairs in the batch."""
    x = batch["x"].to(device)
    pairs = batch["pairs"].to(device)

    if name == "mlp_semantic":
        return model(
            x=x,
            edge_label_index=pairs,
            object_labels=batch["object_labels"].to(device),
        )

    if name in ("mlp", "mlp_geometry"):
        return model(x=x, edge_label_index=pairs)

    raise ValueError(f"Unsupported model: {name}")


@torch.no_grad()
def evaluate_graph_batch(model, graphs, name, device, compute_loss=False):
    """Return image-level results and optional validation CE statistics."""
    batch, slices = collate_evaluation_graphs(graphs)
    logits = forward_evaluation_batch(model, batch, name, device)

    loss_sum = 0.0
    total_pairs = 0

    if compute_loss:
        targets = torch.cat([g.edge_label for g in graphs]).to(device)
        mask = targets >= 0

        if mask.any():
            loss_sum = torch.nn.functional.cross_entropy(
                logits[mask],
                targets[mask],
                reduction="sum",
            ).item()
            total_pairs = int(mask.sum().item())

    # Transfer predictions once per batch. Image-level metric logic
    # remains on CPU, exactly as in the existing evaluator.
    logits_cpu = logits.cpu()
    results = []

    for graph, (begin, end) in zip(graphs, slices, strict=True):
        results.append(
            evaluate_image(
                logits=logits_cpu[begin:end],
                candidate_pairs=graph.edge_label_index,
                gt_pairs=graph.gt_relation_index,
                gt_predicates=graph.gt_predicate,
            )
        )

    return results, loss_sum, total_pairs
