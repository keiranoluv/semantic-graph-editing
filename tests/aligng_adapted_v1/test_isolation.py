import copy

import pytest
import torch

from experiments.aligng_adapted_v1.batching import collate, model_inputs

from .conftest import graph, model


@pytest.mark.parametrize("classes", [51, 311])
@pytest.mark.parametrize("training", [False, True])
def test_alone_batch_permutation_and_other_image_mutation(classes, training):
    net = model(classes)
    net.train(training)  # dropout=0 also proves no hidden train-mode BatchNorm dependence.
    a, b = graph(3), graph(5, (7,), seed=83)
    empty, single = graph(0, ()), graph(1, ())
    alone = net(**model_inputs(collate([a], classes)))
    both = net(**model_inputs(collate([a, empty, b, single], classes)))
    permuted = net(**model_inputs(collate([single, b, empty, a], classes)))
    torch.testing.assert_close(alone, both[:6], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(alone, permuted[-6:], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(both[6:], permuted[:-6], atol=1e-6, rtol=1e-6)
    changed = copy.deepcopy(b)
    changed.x = changed.x * 30 + 14
    changed.object_labels = torch.ones_like(changed.object_labels)
    mutated = net(**model_inputs(collate([a, changed], classes)))
    torch.testing.assert_close(alone, mutated[:6], atol=1e-6, rtol=1e-6)
    assert not torch.allclose(both[6:], mutated[6:])
    assert not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in net.modules())


def test_query_selection_and_labels_cannot_change_context_or_image_prototypes():
    net = model(dropout=0.2).train()
    a, altered = graph(6), graph(6, (9, 11))
    b1 = collate([a], 51, torch.Generator().manual_seed(42))
    b2 = collate([altered], 51, torch.Generator().manual_seed(123), negative_ratio=0)
    captured = []
    handle = net.feedback.register_forward_hook(lambda module, args, out: captured.append(out[1].detach().clone()))
    for b in (b1, b2):
        torch.manual_seed(100)
        net(**model_inputs(b))
    handle.remove()
    torch.testing.assert_close(captured[0], captured[1], atol=0, rtol=0)
    # Even invalid/NaN target contents are excluded from model forward.
    net.eval()
    expected = net(**model_inputs(b1))
    b1["targets"].fill_(float("nan"))
    torch.testing.assert_close(net(**model_inputs(b1)), expected, atol=0, rtol=0)
    with pytest.raises(TypeError):
        net(**model_inputs(b1), targets=b1["targets"])


def test_query_reordering_repetition_context_permutation_and_direction():
    net = model().eval()
    batch = collate([graph(4)], 51)
    full = net(**model_inputs(batch))
    select = torch.tensor([9, 0, 3, 0])
    batch["query_pairs"] = batch["query_pairs"][:, select]
    batch["query_ptr"] = torch.tensor([0, 4])
    torch.testing.assert_close(net(**model_inputs(batch)), full[select], atol=1e-6, rtol=1e-6)
    # (0,1) and (1,0) must remain distinguishable.
    assert not torch.allclose(full[0], full[3])
    batch["context_pairs"] = batch["context_pairs"][:, torch.randperm(12)]
    torch.testing.assert_close(net(**model_inputs(batch)), full[select], atol=2e-6, rtol=2e-6)


def test_no_persistent_adapted_prototype_state_and_eval_determinism():
    net = model(dropout=0.2).eval()
    state = copy.deepcopy(net.state_dict())
    a, b = collate([graph()], 51), collate([graph(7, seed=100)], 51)
    first = net(**model_inputs(a))
    net(**model_inputs(b))
    torch.testing.assert_close(net(**model_inputs(a)), first, atol=0, rtol=0)
    for name, tensor in net.state_dict().items():
        torch.testing.assert_close(tensor, state[name], atol=0, rtol=0)


@pytest.mark.parametrize(
    "bad",
    [
        "cross_context",
        "cross_query",
        "pointer_end",
        "pointer_order",
        "pointer_count",
        "pointer_dtype",
        "pair_shape",
        "labels",
        "nan_features",
    ],
)
def test_malformed_batch_fails_before_cross_image_attention(bad):
    net = model()
    batch = collate([graph(3), graph(4)], 51)
    if bad == "cross_context":
        batch["context_pairs"][1, 0] = 3
    elif bad == "cross_query":
        batch["query_pairs"][1, 0] = 3
    elif bad == "pointer_end":
        batch["query_ptr"][-1] -= 1
    elif bad == "pointer_order":
        batch["node_ptr"][1] = 8
    elif bad == "pointer_count":
        batch["node_ptr"] = torch.tensor([0, 7])
    elif bad == "pointer_dtype":
        batch["context_ptr"] = batch["context_ptr"].float()
    elif bad == "pair_shape":
        batch["query_pairs"] = batch["query_pairs"].T
    elif bad == "labels":
        batch["object_labels"][0] = 8
    else:
        batch["x"][0, 0] = float("nan")
    with pytest.raises(ValueError):
        net(**model_inputs(batch))
