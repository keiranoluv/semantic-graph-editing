"""Regression tests for MLP graph mini-batching."""

import torch
from torch_geometric.data import Data

from scripts.evaluate_mlp import build_model
from scripts.pilot_mlp import collate_training_graphs, forward_batch


def make_graph(n, pairs, labels, object_class=1):
    return Data(
        x=torch.randn(n, 5),
        object_labels=torch.full((n,), object_class, dtype=torch.long),
        edge_label_index=torch.tensor(pairs, dtype=torch.long),
        edge_label=torch.tensor(labels, dtype=torch.long),
        num_nodes=n,
    )


def test_pair_indices_are_offset_correctly():
    a = make_graph(2, [[0], [1]], [1])
    b = make_graph(3, [[0], [2]], [2])

    batch, skipped = collate_training_graphs([a, b], 0, torch.Generator().manual_seed(42))

    assert skipped == 0
    assert batch["x"].shape == (5, 5)
    assert batch["pairs"].tolist() == [[0, 2], [1, 4]]
    assert batch["targets"].tolist() == [1, 2]


def test_skipped_graph_does_not_shift_indices():
    a = make_graph(2, [[0], [1]], [1])
    skipped_graph = make_graph(4, [[0], [1]], [0])
    b = make_graph(3, [[0], [2]], [2])

    batch, skipped = collate_training_graphs(
        [a, skipped_graph, b],
        0,
        torch.Generator().manual_seed(42),
    )

    assert skipped == 1
    assert batch["x"].shape == (5, 5)
    assert batch["pairs"].tolist() == [[0, 2], [1, 4]]


def test_geometry_forward_batch():
    graph = make_graph(2, [[0], [1]], [1])
    batch, _ = collate_training_graphs([graph], 0, torch.Generator().manual_seed(42))
    model = build_model(
        {
            "model": "mlp",
            "hidden_dim": 16,
            "num_classes": 51,
        }
    )

    logits = forward_batch(model, batch, "mlp", torch.device("cpu"))
    assert logits.shape == (1, 51)


def test_semantic_forward_batch():
    graph = make_graph(2, [[0], [1]], [1])
    batch, _ = collate_training_graphs([graph], 0, torch.Generator().manual_seed(42))
    model = build_model(
        {
            "model": "mlp_semantic",
            "hidden_dim": 16,
            "embedding_dim": 8,
            "num_object_classes": 151,
            "num_classes": 51,
        }
    )

    logits = forward_batch(model, batch, "mlp_semantic", torch.device("cpu"))
    assert logits.shape == (1, 51)
