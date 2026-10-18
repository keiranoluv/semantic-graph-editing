"""Synthetic tests for the MLP training pipeline."""

import torch
from torch_geometric.data import Data

from src.models.mlp import MLPPredicateClassifier
from src.training.trainer import train_step


def make_synthetic_graph() -> Data:
    """Create a small, fixed graph for training tests."""
    x = torch.eye(4, 5, dtype=torch.float32)

    pairs = [(src, dst) for src in range(4) for dst in range(4) if src != dst]

    edge_label_index = torch.tensor(
        pairs,
        dtype=torch.long,
    ).T.contiguous()

    labels = torch.zeros(len(pairs), dtype=torch.long)

    labels[pairs.index((0, 1))] = 1
    labels[pairs.index((2, 3))] = 2

    return Data(
        x=x,
        edge_label_index=edge_label_index,
        edge_label=labels,
        num_nodes=4,
    )


def test_train_step_updates_parameters():
    torch.manual_seed(42)

    graph = make_synthetic_graph()
    model = MLPPredicateClassifier(
        input_dim=5,
        hidden_dim=32,
        num_classes=3,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    before = [p.detach().clone() for p in model.parameters()]

    loss = train_step(model, graph, optimizer)

    assert loss is not None
    assert torch.isfinite(torch.tensor(loss))
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters(), strict=False))


def test_empty_positive_graph_skipped():
    graph = make_synthetic_graph()
    graph.edge_label.zero_()

    model = MLPPredicateClassifier(
        input_dim=5,
        hidden_dim=32,
        num_classes=3,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    before = [p.detach().clone() for p in model.parameters()]

    loss = train_step(model, graph, optimizer)

    assert loss is None
    assert all(torch.equal(old, new) for old, new in zip(before, model.parameters(), strict=False))


def test_synthetic_overfitting():
    torch.manual_seed(42)
    torch.set_num_threads(1)

    graph = make_synthetic_graph()

    model = MLPPredicateClassifier(
        input_dim=5,
        hidden_dim=64,
        num_classes=3,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.03,
        weight_decay=0.0,
    )

    losses = []

    # 12 candidate pairs: 2 positives + 10 backgrounds.
    # Ratio 10 includes every background pair.
    for _ in range(200):
        loss = train_step(
            model,
            graph,
            optimizer,
            negative_ratio=10,
        )
        assert loss is not None
        losses.append(loss)

    model.eval()

    with torch.no_grad():
        logits = model(graph.x, graph.edge_label_index)
        predictions = logits.argmax(dim=-1)

    accuracy = (predictions == graph.edge_label).float().mean().item()

    assert losses[-1] < losses[0]
    assert losses[-1] < 0.05
    assert accuracy == 1.0
