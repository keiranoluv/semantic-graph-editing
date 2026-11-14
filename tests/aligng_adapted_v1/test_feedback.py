import copy

import pytest
import torch

from experiments.aligng_adapted_v1.feedback import PrototypeFeedback
from experiments.aligng_adapted_v1.model import AlignGConfig

from .upstream_reference import SGGLayer


@pytest.mark.parametrize("classes", [51, 311])
@pytest.mark.parametrize("chunk", [3, 1024])
def test_pinned_upstream_values_and_all_gradients(classes, chunk):
    reference = SGGLayer(classes, 8, 2, dropout=0).double()
    actual = PrototypeFeedback(8, 2, dropout=0, query_chunk_size=chunk).double()
    actual.load_state_dict(reference.state_dict(), strict=True)
    p = torch.randn(classes, 8, dtype=torch.double, requires_grad=True)
    h = torch.randn(5, 8, dtype=torch.double, requires_grad=True)
    expected = reference(p, h, [torch.zeros(5, 2, dtype=torch.long)])
    observed, adapted = actual(p, h, h)
    torch.testing.assert_close(observed, expected, atol=2e-12, rtol=2e-12)
    assert adapted.shape == (classes, 8) and torch.isfinite(adapted).all()
    ga = torch.autograd.grad(observed.square().sum(), (p, h, *actual.parameters()), retain_graph=True)
    ge = torch.autograd.grad(expected.square().sum(), (p, h, *reference.parameters()))
    for a, e in zip(ga, ge, strict=True):
        torch.testing.assert_close(a, e, atol=1e-10, rtol=1e-9)


def test_pinned_upstream_dropout_and_checkpoint_rng():
    reference = SGGLayer(5, 8, 2, dropout=0.2).double().train()
    actual = PrototypeFeedback(8, 2, dropout=0.2, query_chunk_size=1024).double().train()
    actual.load_state_dict(reference.state_dict())
    p, h = torch.randn(5, 8, dtype=torch.double), torch.randn(7, 8, dtype=torch.double)
    torch.manual_seed(91)
    expected = reference(p, h, [torch.zeros(7, 2, dtype=torch.long)])
    torch.manual_seed(91)
    observed, _ = actual(p, h, h)
    torch.testing.assert_close(observed, expected, atol=0, rtol=0)
    ga = torch.autograd.grad(observed.square().sum(), tuple(actual.parameters()))
    ge = torch.autograd.grad(expected.square().sum(), tuple(reference.parameters()))
    for a, e in zip(ga, ge, strict=True):
        torch.testing.assert_close(a, e, atol=1e-12, rtol=1e-12)


def test_attention_normalizes_over_all_keys_including_chunk_boundary():
    layer = PrototypeFeedback(4, 2, 0, query_chunk_size=1).double()
    q = torch.zeros(3, 4, dtype=torch.double)
    k = torch.randn(7, 4, dtype=torch.double)
    v = torch.arange(28, dtype=torch.double).reshape(7, 4)
    torch.testing.assert_close(layer.attend(q, k, v), v.mean(0).expand(3, -1), atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(layer.attend(q, k, torch.ones_like(v)), torch.ones_like(q))
    q = torch.randn_like(q) * 10
    expected = []
    for row in q:
        head_values = []
        for head in range(2):
            section = slice(head * 2, (head + 1) * 2)
            weights = (k[:, section] @ row[section] / 2**0.5).softmax(0)
            assert weights.sum().item() == pytest.approx(1)
            head_values.append(weights @ v[:, section])
        expected.append(torch.cat(head_values))
    torch.testing.assert_close(layer.attend(q, k, v), torch.stack(expected), atol=1e-12, rtol=1e-12)


def test_query_subset_does_not_change_prototypes_and_preserves_order():
    layer = PrototypeFeedback(8, 2, 0).eval()
    p, h = torch.randn(51, 8), torch.randn(9, 8)
    full, full_p = layer(p, h, h)
    ids = torch.tensor([7, 1, 7, 0])
    selected, selected_p = layer(p, h, h[ids])
    torch.testing.assert_close(selected_p, full_p, atol=0, rtol=0)
    torch.testing.assert_close(selected, full[ids], atol=1e-6, rtol=1e-6)
    _, other_p = layer(p, h + torch.randn_like(h), h)
    assert not torch.allclose(other_p, full_p)


@pytest.mark.parametrize("empty_context,empty_query", [(True, True), (True, False), (False, True)])
def test_empty_feedback_has_explicit_finite_fallback(empty_context, empty_query):
    layer = PrototypeFeedback(8, 2, 0)
    p = torch.randn(51, 8)
    context = torch.randn(0 if empty_context else 4, 8)
    queries = torch.randn(0 if empty_query else 3, 8)
    out, adapted = layer(p, context, queries)
    assert out.shape == queries.shape and torch.isfinite(out).all() and torch.isfinite(adapted).all()
    if empty_context:
        torch.testing.assert_close(adapted, layer.proto_norm(p), rtol=0, atol=0)


def test_reverse_only_and_disabled_reverse_are_explicit():
    layer = PrototypeFeedback(8, 2, 0).eval()
    p, h = torch.randn(51, 8), torch.randn(7, 8)
    actual, adapted = layer(p, h, h, prototype_update_enabled=False)
    other, _ = layer(p, h * 100, h, prototype_update_enabled=False)
    torch.testing.assert_close(adapted, layer.proto_norm(p))
    torch.testing.assert_close(actual, other, rtol=0, atol=0)
    out, _ = layer(p, h, h, reverse_feedback_enabled=False)
    assert out is h


def test_checkpointed_chunks_match_values_gradients_and_dropout_rng():
    a = PrototypeFeedback(8, 2, 0.2, query_chunk_size=2, checkpoint_chunks=True).double()
    b = copy.deepcopy(a)
    b.checkpoint_chunks = False
    p, h = torch.randn(5, 8, dtype=torch.double), torch.randn(7, 8, dtype=torch.double)
    torch.manual_seed(13)
    oa, _ = a(p, h, h)
    rng_a = torch.get_rng_state()
    ga = torch.autograd.grad(oa.square().sum(), tuple(a.parameters()))
    assert torch.equal(rng_a, torch.get_rng_state())
    torch.manual_seed(13)
    ob, _ = b(p, h, h)
    gb = torch.autograd.grad(ob.square().sum(), tuple(b.parameters()))
    torch.testing.assert_close(oa, ob, atol=0, rtol=0)
    for x, y in zip(ga, gb, strict=True):
        torch.testing.assert_close(x, y, atol=0, rtol=0)


@pytest.mark.parametrize(
    "flags",
    [
        {"prototype_update_mode": "ema"},
        {"loss_mode": "bce"},
        {"num_attention_heads": 0},
        {"pair_chunk_size": 0},
        {"feedback_enabled": "false"},
    ],
)
def test_unsupported_config_rejected(flags):
    with pytest.raises(ValueError):
        AlignGConfig(**flags)


def test_invalid_dimensions_and_nonfinite_feedback_rejected():
    with pytest.raises(ValueError):
        PrototypeFeedback(7, 2)
    layer = PrototypeFeedback(8, 2)
    with pytest.raises(ValueError):
        layer(torch.full((51, 8), float("nan")), torch.zeros(2, 8), torch.zeros(2, 8))
