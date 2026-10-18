"""Batched inference must match per-image inference."""

import pytest
import torch
from torch_geometric.data import Data

from scripts.evaluate_mlp import build_model, predict
from src.evaluation.batched import collate_evaluation_graphs, evaluate_graph_batch, forward_evaluation_batch
from src.evaluation.scene_graph import evaluate_image, summarize_evaluation


def make_graph(n, predicate):
    pairs = torch.tensor(
        [[i for i in range(n) for j in range(n) if i != j], [j for i in range(n) for j in range(n) if i != j]],
        dtype=torch.long,
    )
    labels = torch.zeros(pairs.shape[1], dtype=torch.long)
    labels[0] = predicate

    return Data(
        x=torch.randn(n, 5),
        object_labels=torch.arange(1, n + 1, dtype=torch.long),
        edge_label_index=pairs,
        edge_label=labels,
        gt_relation_index=pairs[:, :1].clone(),
        gt_predicate=torch.tensor([predicate]),
        num_nodes=n,
    )


@pytest.mark.parametrize("name", ["mlp", "mlp_semantic"])
def test_batched_logits_equal_individual_logits(name):
    torch.manual_seed(42)

    config = {
        "model": name,
        "hidden_dim": 32,
        "num_classes": 51,
        "embedding_dim": 8,
        "num_object_classes": 151,
    }
    model = build_model(config).eval()
    graphs = [make_graph(3, 1), make_graph(4, 2)]

    with torch.no_grad():
        reference = torch.cat(
            [predict(model, graph, name) for graph in graphs],
            dim=0,
        )
        batch, _ = collate_evaluation_graphs(graphs)
        actual = forward_evaluation_batch(model, batch, name, torch.device("cpu"))

    torch.testing.assert_close(actual, reference)


@pytest.mark.parametrize("name", ["mlp", "mlp_semantic"])
def test_batched_metrics_equal_individual_metrics(name):
    torch.manual_seed(42)

    model = build_model(
        {
            "model": name,
            "hidden_dim": 32,
            "num_classes": 51,
            "embedding_dim": 8,
            "num_object_classes": 151,
        }
    ).eval()

    graphs = [make_graph(3, 1), make_graph(4, 2)]

    with torch.no_grad():
        reference = [
            evaluate_image(
                logits=predict(model, graph, name),
                candidate_pairs=graph.edge_label_index,
                gt_pairs=graph.gt_relation_index,
                gt_predicates=graph.gt_predicate,
            )
            for graph in graphs
        ]

        actual, _, _ = evaluate_graph_batch(model, graphs, name, torch.device("cpu"))

    expected_metrics = summarize_evaluation(reference)
    actual_metrics = summarize_evaluation(actual)

    for key in ("R@50", "R@100", "mR@50", "mR@100"):
        assert actual_metrics[key] == pytest.approx(expected_metrics[key], abs=1e-6)


def test_batched_validation_cross_entropy():
    torch.manual_seed(42)

    model = build_model(
        {
            "model": "mlp",
            "hidden_dim": 16,
            "num_classes": 51,
        }
    ).eval()
    graphs = [make_graph(3, 1), make_graph(4, 2)]

    with torch.no_grad():
        results, loss_sum, pairs = evaluate_graph_batch(
            model,
            graphs,
            "mlp",
            torch.device("cpu"),
            compute_loss=True,
        )

    assert len(results) == 2
    assert pairs == sum(g.edge_label.numel() for g in graphs)
    assert loss_sum > 0
