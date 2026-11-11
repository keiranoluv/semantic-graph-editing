"""Numerical/gradient reference for PE-Net code losses and adapter edge cases.

Reference formulas adapted from VL-Group/PENET@9c9f507; MIT, Copyright (c) 2018 Facebook.
See licenses/PENET-MIT.txt. Small synthetic fixtures are never experiment data.
"""

import pytest
import torch
import torch.nn.functional as F

from src.models.penet import PENetPredicateClassifier
from src.training.penet import prototype_losses


def upstream_reference(logits, relation, proto, labels):
    # Literal broadcast/sort loss behavior of the pinned 51-class implementation.
    unit = proto / proto.norm(dim=1, keepdim=True)
    sim = unit @ unit.detach().T
    cosine = torch.norm(torch.norm(sim, p=2, dim=1), p=1) / (51 * 51)
    distance = (proto[:, None, :] - proto.detach()[None, :, :]).norm(dim=2) ** 2
    nearest = distance.sort(dim=1).values[:, :2].sum(1)
    separation = torch.maximum(torch.zeros_like(nearest), 7 - nearest).mean()
    dist = (relation[:, None, :] - proto[None, :, :]).norm(dim=2) ** 2
    mask = torch.ones_like(dist)
    mask[torch.arange(len(labels)), labels] = 0
    negatives = (dist * mask).sort(dim=1).values[:, :11].sum(1) / 10
    positives = dist[torch.arange(len(labels)), labels]
    margin = torch.maximum(torch.zeros_like(positives), positives - negatives + 1).mean()
    return {
        "ce": F.cross_entropy(logits, labels),
        "pl_euclidean": margin,
        "pr_cosine": cosine,
        "pr_euclidean": separation,
    }


def test_51_class_values_and_gradients_match_pinned_code():
    torch.manual_seed(9)
    logits = torch.randn(7, 51, dtype=torch.float64, requires_grad=True)
    relation = (torch.randn(7, 8, dtype=torch.float64) * 0.1).requires_grad_()
    proto = (torch.randn(51, 8, dtype=torch.float64) * 0.1).requires_grad_()
    labels = torch.tensor([0, 1, 2, 7, 12, 33, 50])
    target = F.one_hot(labels, 51).double()
    actual = prototype_losses(logits, relation, proto, target)
    expected = upstream_reference(logits, relation, proto, labels)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=1e-10, atol=1e-10)
    ga = torch.autograd.grad(actual["total"], (logits, relation, proto), retain_graph=True)
    ge = torch.autograd.grad(sum(expected.values()), (logits, relation, proto))
    for a, e in zip(ga, ge, strict=True):
        torch.testing.assert_close(a, e, rtol=1e-9, atol=1e-10)


def test_all_known_positives_are_excluded_from_hard_negatives():
    relation = torch.tensor([[0.0]], requires_grad=True)
    proto = torch.tensor([[5.0], [0.0], [1.0], [-2.0]], requires_grad=True)
    target = torch.tensor([[0.0, 0.5, 0.5, 0.0]])
    losses = prototype_losses(torch.zeros(1, 4), relation, proto, target, mode="pl", k1=1, gamma1=4.0)
    assert losses["pl_euclidean"].item() == pytest.approx(0.5)
    losses["total"].backward()
    assert torch.isfinite(proto.grad).all()


def test_no_negative_zero_mask_leak_when_fewer_than_k_negatives():
    losses = prototype_losses(
        torch.zeros(1, 3),
        torch.tensor([[0.0]]),
        torch.tensor([[2.0], [0.0], [1.0]]),
        torch.tensor([[0.0, 0.5, 0.5]]),
        mode="pl",
        k1=10,
        gamma1=4,
    )
    assert losses["pl_euclidean"].item() == pytest.approx(0.5)


def test_empty_pairs_do_not_create_prototype_regularizer_update():
    logits = torch.empty(0, 311, requires_grad=True)
    proto = torch.randn(311, 8, requires_grad=True)
    losses = prototype_losses(logits, torch.empty(0, 8), proto, torch.empty(0, 311))
    assert all(x.item() == 0 for x in losses.values())
    losses["total"].backward()
    assert proto.grad is None


@pytest.mark.parametrize(
    "bad", [torch.tensor([[0.5, 0.5, 0.0]]), torch.tensor([[0.0, 0.0, 0.0]]), torch.tensor([[0.0, float("nan"), 1.0]])]
)
def test_invalid_supervision_rejected(bad):
    with pytest.raises(ValueError):
        prototype_losses(torch.zeros(1, 3), torch.zeros(1, 2), torch.zeros(3, 2), bad)


@pytest.mark.parametrize("mode", ["ce", "pl", "plpr"])
def test_disabled_components_are_zero(mode):
    torch.manual_seed(5)
    out = prototype_losses(
        torch.randn(3, 51),
        torch.randn(3, 8),
        torch.randn(51, 8),
        F.one_hot(torch.tensor([0, 1, 50]), 51).float(),
        mode=mode,
    )
    if mode == "ce":
        assert out["pl_euclidean"].item() == 0
    if mode != "plpr":
        assert out["pr_cosine"].item() == out["pr_euclidean"].item() == 0
    assert torch.isfinite(out["total"])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("classes", [51, 311])
def test_model_forward_empty_batch_equivalence_direction_and_gradients(device, classes):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(3)
    model = PENetPredicateClassifier(torch.randn(1700, 8), torch.randn(classes, 8), semantic_dim=16, dropout=0).to(
        device
    )
    x = torch.randn(3, 517, device=device)
    labels = torch.tensor([1699, 1, 2], device=device)
    pairs = torch.tensor([[0, 1, 1], [1, 0, 2]], device=device)
    logits, r, p = model(x, pairs, labels, return_embeddings=True)
    assert logits.shape == (3, classes) and not torch.allclose(logits[0], logits[1])
    each = torch.cat([model(x, pairs[:, i : i + 1], labels) for i in range(3)])
    torch.testing.assert_close(each, logits, rtol=1e-5, atol=1e-5)
    targets = F.one_hot(torch.tensor([0, classes - 1, 1], device=device), classes).float()
    prototype_losses(logits, r, p, targets)["total"].backward()
    assert model.object_embedding.weight.grad is not None and x.grad is None
    assert all(t.grad is None or torch.isfinite(t.grad).all() for t in model.parameters())
    for n in [0, 1]:
        empty = model(x[:n], torch.empty(2, 0, dtype=torch.long, device=device), labels[:n])
        assert empty.shape == (0, classes)
