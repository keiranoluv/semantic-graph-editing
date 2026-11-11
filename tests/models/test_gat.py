"""GAT model contracts, shape, message passing, and gradient flow."""

import pytest
import torch

pytest.importorskip("torch_geometric")

from src.models.gat import GATPredicateClassifier, build_gat_model


def sample():
    x = torch.rand(4, 5)
    classes = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    edges = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3], [1, 2, 0, 3, 0, 3, 1, 2]])
    pairs = torch.tensor([[0, 1, 2], [1, 3, 0]])
    return x, classes, edges, pairs


def model():
    return GATPredicateClassifier(hidden_dim=32, num_heads=4, num_layers=2, dropout=0.0)


def test_output_shape_and_gradient():
    torch.manual_seed(42)
    net = model()
    x, labels, edges, pairs = sample()
    logits = net(x, edges, pairs, labels)
    assert logits.shape == (3, 51)
    assert torch.isfinite(logits).all()
    logits.square().mean().backward()
    assert net.object_embedding.weight.grad is not None
    assert net.object_embedding.weight.grad.abs().sum() > 0
    assert net.convs[0].lin.weight.grad is not None
    assert net.convs[0].lin.weight.grad.abs().sum() > 0


def test_topology_affects_output():
    torch.manual_seed(42)
    net = model().eval()
    x, labels, edges, pairs = sample()
    with torch.no_grad():
        connected = net(x, edges, pairs, labels)
        # No inter-object edges. Self-loops are added internally by GATConv.
        isolated = net(x, edges[:, :0], pairs, labels)
    assert not torch.allclose(connected, isolated)


def test_ordered_pairs_and_zero_pairs():
    net = model().eval()
    x, labels, edges, pairs = sample()
    with torch.no_grad():
        original = net(x, edges, pairs, labels)
        reverse = net(x, edges, pairs[:, [2, 0, 1]], labels)
        zero = net(x, edges, pairs[:, :0], labels)
    torch.testing.assert_close(reverse, original[[2, 0, 1]])
    assert zero.shape == (0, 51)


def test_rejects_invalid_object_class():
    net = model()
    x, labels, edges, pairs = sample()
    labels[0] = 999
    with pytest.raises(ValueError, match="vocabulary"):
        net(x, edges, pairs, labels)


def test_config_roundtrip():
    config = {
        "model": "gat",
        "geometry_dim": 5,
        "num_object_classes": 151,
        "embedding_dim": 32,
        "hidden_dim": 32,
        "num_classes": 51,
        "num_heads": 4,
        "num_layers": 2,
        "dropout": 0.0,
    }
    first = build_gat_model(config)
    second = build_gat_model(config)
    second.load_state_dict(first.state_dict())
    x, labels, edges, pairs = sample()
    first.eval()
    second.eval()
    with torch.no_grad():
        torch.testing.assert_close(first(x, edges, pairs, labels), second(x, edges, pairs, labels))
