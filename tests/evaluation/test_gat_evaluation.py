"""Batched GAT evaluation agrees with single-image inference and rankings."""

import pytest
import torch
from torch_geometric.data import Data

from src.evaluation.gat import evaluate_gat_batch
from src.evaluation.scene_graph import evaluate_image, summarize_evaluation
from src.graph.gnn_batching import collate_gnn_graphs, forward_gat_batch
from src.models.gat import GATPredicateClassifier


def make_graph(n, predicate):
    src = [i for i in range(n) for j in range(n) if i != j]
    dst = [j for i in range(n) for j in range(n) if i != j]
    pairs = torch.tensor([src, dst], dtype=torch.long)
    targets = torch.zeros(pairs.shape[1], dtype=torch.long)
    targets[0] = predicate
    return Data(
        x=torch.rand(n, 5),
        object_labels=torch.arange(1, n + 1, dtype=torch.long),
        edge_index=pairs,
        edge_label_index=pairs.clone(),
        edge_label=targets,
        gt_relation_index=pairs[:, :1].clone(),
        gt_predicate=torch.tensor([predicate]),
        num_nodes=n,
    )


def test_batched_forward_and_metrics_match_images():
    torch.manual_seed(42)
    model = GATPredicateClassifier(hidden_dim=32, num_heads=4, dropout=0.0).eval()
    graphs = [make_graph(3, 1), make_graph(5, 2)]

    with torch.no_grad():
        originals = [model(g.x, g.edge_index, g.edge_label_index, g.object_labels) for g in graphs]
        batch, slices, _ = collate_gnn_graphs(graphs)
        combined = forward_gat_batch(model, batch, torch.device("cpu"))
        for (start, end), expected in zip(slices, originals, strict=True):
            torch.testing.assert_close(combined[start:end], expected, rtol=1e-5, atol=1e-6)
        reference = [
            evaluate_image(original, g.edge_label_index, g.gt_relation_index, g.gt_predicate)
            for g, original in zip(graphs, originals, strict=True)
        ]
        actual, loss_sum, pair_count = evaluate_gat_batch(model, graphs, torch.device("cpu"), compute_loss=True)

    assert loss_sum > 0
    assert pair_count == sum((g.edge_label >= 0).sum().item() for g in graphs)
    expected_metrics = summarize_evaluation(reference)
    actual_metrics = summarize_evaluation(actual)
    for key in ("R@50", "R@100", "mR@50", "mR@100"):
        assert actual_metrics[key] == pytest.approx(expected_metrics[key], abs=1e-7)


def test_background_and_ambiguous_labels_do_not_change_model_input():
    model = GATPredicateClassifier(hidden_dim=32, num_heads=4, dropout=0.0).eval()
    g = make_graph(3, 1)
    with torch.no_grad():
        original = model(g.x, g.edge_index, g.edge_label_index, g.object_labels)
        g.edge_label.fill_(-1)  # Supervision changed; graph inputs unchanged.
        updated = model(g.x, g.edge_index, g.edge_label_index, g.object_labels)
    torch.testing.assert_close(original, updated)
