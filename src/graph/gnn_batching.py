"""Batch independent scene graphs while preserving full message-passing edges."""

import torch

from src.training.sampling import sample_training_pairs


def collate_gnn_graphs(graphs, *, training=False, negative_ratio=3, generator=None):
    """Return (batch, pair_slices, skipped_graphs).

    In training mode, sample ONLY supervision pairs; edge_index retains all
    edges for every included image. Skip images with no single-label positives,
    just as the MLP pilot runner does. All index offsets use INCLUDED graphs.
    During evaluation every candidate pair is retained.
    """
    xs, object_labels, message_edges, pairs, targets = [], [], [], [], []
    slices = []
    node_offset = 0
    pair_offset = 0
    skipped = 0

    for graph in graphs:
        if training:
            selected = sample_training_pairs(graph.edge_label, negative_ratio=negative_ratio, generator=generator)
            if selected.numel() == 0:
                skipped += 1
                continue
        else:
            selected = torch.arange(graph.edge_label_index.shape[1], dtype=torch.long)

        num_nodes = int(graph.num_nodes)
        xs.append(graph.x)
        object_labels.append(graph.object_labels)
        message_edges.append(graph.edge_index + node_offset)
        pairs.append(graph.edge_label_index[:, selected] + node_offset)
        targets.append(graph.edge_label[selected])

        count = selected.numel()
        slices.append((pair_offset, pair_offset + count))
        pair_offset += count
        node_offset += num_nodes

    if not xs:
        return None, slices, skipped

    return (
        {
            "x": torch.cat(xs, dim=0),
            "object_labels": torch.cat(object_labels, dim=0),
            "edge_index": torch.cat(message_edges, dim=1),
            "pairs": torch.cat(pairs, dim=1),
            "targets": torch.cat(targets, dim=0),
        },
        slices,
        skipped,
    )


def forward_gat_batch(model, batch, device):
    """Execute the same forward interface in both training and evaluation."""
    return model(
        x=batch["x"].to(device),
        edge_index=batch["edge_index"].to(device),
        edge_label_index=batch["pairs"].to(device),
        object_labels=batch["object_labels"].to(device),
    )
