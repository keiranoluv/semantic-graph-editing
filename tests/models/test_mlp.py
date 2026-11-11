"""Synthetic tests for the non-graph MLP baseline."""

import torch

from src.models.mlp import MLPPredicateClassifier


def test_forward_shape():
    model = MLPPredicateClassifier(input_dim=5)

    x = torch.randn(3, 5)
    pairs = torch.tensor([[0, 0, 1], [1, 2, 2]])

    logits = model(x, pairs)

    assert logits.shape == (3, 51)
    assert torch.isfinite(logits).all()


def test_backward():
    model = MLPPredicateClassifier(input_dim=5)

    x = torch.randn(3, 5)
    pairs = torch.tensor([[0, 1], [1, 2]])

    loss = model(x, pairs).sum()
    loss.backward()

    assert all(p.grad is not None for p in model.parameters())


def test_empty_candidates():
    model = MLPPredicateClassifier(input_dim=5)

    x = torch.randn(3, 5)
    pairs = torch.empty((2, 0), dtype=torch.long)

    logits = model(x, pairs)

    assert logits.shape == (0, 51)


def test_candidate_order_preserved():
    model = MLPPredicateClassifier(input_dim=5)
    model.eval()

    x = torch.randn(3, 5)

    pairs = torch.tensor([[0, 1], [1, 2]])
    reversed_pairs = pairs.flip(1)

    with torch.no_grad():
        original = model(x, pairs)
        reversed_output = model(x, reversed_pairs)

    torch.testing.assert_close(
        original,
        reversed_output.flip(0),
    )


def test_invalid_feature_dimension():
    model = MLPPredicateClassifier(input_dim=5)

    x = torch.randn(3, 7)
    pairs = torch.tensor([[0], [1]])

    try:
        model(x, pairs)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")
