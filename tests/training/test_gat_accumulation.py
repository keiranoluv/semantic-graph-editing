"""Accumulated small-batch CE matches one equivalent graph mega-batch update."""

import torch
from torch_geometric.data import Data

from scripts.train_gat import train_one_epoch
from src.graph.gnn_batching import collate_gnn_graphs, forward_gat_batch
from src.models.gat import GATPredicateClassifier


def graph(n, predicate):
    src = [i for i in range(n) for j in range(n) if i != j]
    dst = [j for i in range(n) for j in range(n) if i != j]
    pairs = torch.tensor([src, dst])
    labels = torch.zeros(len(src), dtype=torch.long)
    labels[0] = predicate
    return Data(
        x=torch.rand(n, 5),
        object_labels=torch.arange(1, n + 1),
        edge_index=pairs.clone(),
        edge_label_index=pairs,
        edge_label=labels,
        num_nodes=n,
    )


def test_accumulation_matches_full_batch_gradients():
    torch.manual_seed(17)
    graphs = [graph(3, 1), graph(4, 2), graph(5, 3), graph(3, 4)]
    a = GATPredicateClassifier(hidden_dim=32, num_heads=4, dropout=0.0)
    b = GATPredicateClassifier(hidden_dim=32, num_heads=4, dropout=0.0)
    b.load_state_dict(a.state_dict())
    optimizer_a = torch.optim.SGD(a.parameters(), lr=0.01)
    optimizer_b = torch.optim.SGD(b.parameters(), lr=0.01)

    # One full effective batch of 4 graphs.
    full, _, _ = collate_gnn_graphs(graphs, training=True, negative_ratio=0)
    logits = forward_gat_batch(a, full, torch.device("cpu"))
    loss = torch.nn.functional.cross_entropy(logits, full["targets"])
    loss.backward()
    optimizer_a.step()

    # Two micro-batches of 2 graphs with ONE optimizer step.
    _, optimizer_steps, skipped = train_one_epoch(
        b, graphs, len(graphs), 2, 2, optimizer_b, 0, torch.Generator().manual_seed(42), torch.device("cpu")
    )
    assert optimizer_steps == 1
    assert skipped == 0
    for p_a, p_b in zip(a.parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(p_a, p_b, rtol=1e-4, atol=1e-5)


def test_remainder_accumulation_flushes():
    torch.manual_seed(17)
    graphs = [graph(3, 1), graph(3, 2), graph(3, 3)]
    model = GATPredicateClassifier(hidden_dim=32, num_heads=4, dropout=0.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    _, steps, skipped = train_one_epoch(
        model, graphs, len(graphs), 1, 2, optimizer, 0, torch.Generator().manual_seed(42), torch.device("cpu")
    )
    assert steps == 2
    assert skipped == 0
