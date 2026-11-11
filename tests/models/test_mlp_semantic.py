"""Tests for the semantic MLP predicate baseline."""

import pytest
import torch

from src.models.mlp_semantic import SemanticMLPPredicateClassifier


@pytest.fixture
def sample_inputs():
    x = torch.randn(4, 5)

    object_labels = torch.tensor(
        [1, 2, 3, 4],
        dtype=torch.long,
    )

    pairs = torch.tensor(
        [[0, 0, 1], [1, 2, 3]],
        dtype=torch.long,
    )

    return x, pairs, object_labels


def test_forward_shape(sample_inputs):
    model = SemanticMLPPredicateClassifier()
    x, pairs, labels = sample_inputs

    logits = model(x, pairs, labels)

    assert logits.shape == (3, 51)
    assert torch.isfinite(logits).all()


def test_embedding_receives_gradients(sample_inputs):
    model = SemanticMLPPredicateClassifier()
    x, pairs, labels = sample_inputs

    model(x, pairs, labels).sum().backward()

    grad = model.object_embedding.weight.grad

    assert grad is not None
    assert grad.abs().sum() > 0


def test_semantic_labels_affect_output(sample_inputs):
    torch.manual_seed(42)

    model = SemanticMLPPredicateClassifier()
    model.eval()

    x, pairs, labels = sample_inputs
    changed = labels.clone()
    changed[0] = 5

    with torch.no_grad():
        original = model(x, pairs, labels)
        modified = model(x, pairs, changed)

    assert not torch.allclose(original, modified)


def test_zero_candidate_pairs(sample_inputs):
    model = SemanticMLPPredicateClassifier()
    x, _, labels = sample_inputs

    empty = torch.empty((2, 0), dtype=torch.long)
    logits = model(x, empty, labels)

    assert logits.shape == (0, 51)


def test_invalid_object_labels(sample_inputs):
    model = SemanticMLPPredicateClassifier()
    x, pairs, labels = sample_inputs

    invalid = labels.clone()
    invalid[0] = 999

    with pytest.raises(ValueError, match="vocabulary"):
        model(x, pairs, invalid)


def test_candidate_order_preserved(sample_inputs):
    model = SemanticMLPPredicateClassifier()
    model.eval()

    x, pairs, labels = sample_inputs

    with torch.no_grad():
        original = model(x, pairs, labels)
        reordered = model(x, pairs[:, [2, 0, 1]], labels)

    torch.testing.assert_close(
        reordered,
        original[[2, 0, 1]],
    )
