"""No cross-image edges, no index gaps after skipped images."""

import torch
from torch_geometric.data import Data

from src.graph.gnn_batching import collate_gnn_graphs


def graph(n, labels):
    pairs = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    return Data(
        x=torch.rand(n, 5),
        object_labels=torch.arange(1, n + 1),
        edge_index=pairs.clone(),
        edge_label_index=pairs.clone(),
        edge_label=torch.tensor(labels, dtype=torch.long),
        num_nodes=n,
    )


def test_training_keeps_full_message_passing_topology():
    a = graph(3, [1, 0])
    b = graph(4, [2, 0])
    batch, slices, skipped = collate_gnn_graphs(
        [a, b], training=True, negative_ratio=0, generator=torch.Generator().manual_seed(42)
    )
    assert skipped == 0
    assert batch["x"].shape == (7, 5)
    assert batch["edge_index"].tolist() == [[0, 1, 3, 4], [1, 0, 4, 3]]
    assert batch["pairs"].tolist() == [[0, 3], [1, 4]]
    assert batch["targets"].tolist() == [1, 2]
    assert slices == [(0, 1), (1, 2)]


def test_skipped_graph_does_not_shift_offsets():
    a = graph(2, [1, 0])
    skipped_graph = graph(10, [0, 0])
    b = graph(3, [2, 0])
    batch, _, skipped = collate_gnn_graphs([a, skipped_graph, b], training=True, negative_ratio=0)
    assert skipped == 1
    assert batch["x"].shape[0] == 5
    assert batch["edge_index"].tolist() == [[0, 1, 2, 3], [1, 0, 3, 2]]
    assert batch["pairs"].tolist() == [[0, 2], [1, 3]]


def test_validation_preserves_all_pairs_and_order():
    a = graph(2, [1, -1])
    b = graph(3, [2, 0])
    batch, slices, skipped = collate_gnn_graphs([a, b])
    assert skipped == 0
    assert slices == [(0, 2), (2, 4)]
    assert batch["pairs"].tolist() == [[0, 1, 2, 3], [1, 0, 3, 2]]
    assert batch["targets"].tolist() == [1, -1, 2, 0]


def test_all_skipped_returns_none():
    batch, slices, skipped = collate_gnn_graphs([graph(3, [0, 0])], training=True)
    assert batch is None
    assert slices == []
    assert skipped == 1
