import copy

import pytest
import torch

from experiments.aligng_adapted_v1.batching import collate, model_inputs
from experiments.aligng_adapted_v1.training import batch_loss, train_step
from src.training.penet import prototype_losses

from .conftest import graph, model


@pytest.mark.parametrize("classes", [51, 311])
@pytest.mark.parametrize("mode", ["ce", "pl", "plpr"])
def test_d256_forward_backward_losses_and_frozen_inputs(classes, mode):
    net = model(classes, dim=256, loss_mode=mode)
    batch = collate([graph(3, (1, 1, classes - 1)), graph(4, (), seed=39)], classes)
    batch["x"].requires_grad_()
    outputs = net(**model_inputs(batch), return_embeddings=True)
    losses = batch_loss(net, batch)
    expected = prototype_losses(*outputs, batch["targets"], mode=mode)
    for key in losses:
        torch.testing.assert_close(losses[key], expected[key], atol=0, rtol=0)
        assert torch.isfinite(losses[key])
    losses["total"].backward()
    assert outputs[0].shape == (18, classes)
    assert outputs[1].shape == (18, 512) and outputs[2].shape == (classes, 512)
    assert batch["x"].grad is None
    for name, p in net.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name


@pytest.mark.parametrize("enabled", [True, False])
def test_empty_singleton_skip_optimizer_with_existing_adam_state(enabled):
    net = model(feedback_enabled=enabled)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.2)
    first = train_step(net, collate([graph()], 51), opt)
    assert first["updated"]
    before = copy.deepcopy(net.state_dict())
    opt_before = copy.deepcopy(opt.state_dict())
    batch = collate([graph(0, ()), graph(1, ())], 51)
    result = train_step(net, batch, opt)
    assert not result["updated"] and result["gradient_norm"] is None
    assert all(v.item() == 0 for v in result["losses"].values())
    for name, tensor in net.state_dict().items():
        torch.testing.assert_close(tensor, before[name], atol=0, rtol=0)
    for key, values in opt.state_dict()["state"].items():
        for name, tensor in values.items():
            torch.testing.assert_close(tensor, opt_before["state"][key][name], atol=0, rtol=0)
    assert all(p.grad is None for p in net.parameters())


def test_context_only_pairs_receive_gradient_through_feedback():
    net = model()
    batch = collate([graph(5)], 51, torch.Generator().manual_seed(42), negative_ratio=0)
    captured = []

    def retain_context(module, args):
        args[1].retain_grad()
        captured.append(args[1])

    handle = net.feedback.register_forward_pre_hook(retain_context)
    batch_loss(net, batch)["total"].backward()
    handle.remove()
    assert batch["query_pairs"].shape[1] == 1 and len(captured[0]) == 20
    assert captured[0].grad[1:].abs().sum() > 0


def test_checkpoint_pair_chunks_preserve_dropout_outputs_gradients_and_rng():
    a = model(dropout=0.2, pair_chunk_size=3, attention_query_chunk_size=3)
    b = model(dropout=0.2, pair_chunk_size=3, attention_query_chunk_size=3, checkpoint_chunks=False)
    b.load_state_dict(a.state_dict())
    batch = collate([graph(4)], 51)
    torch.manual_seed(73)
    la = batch_loss(a, batch)["total"]
    rng = torch.get_rng_state()
    la.backward()
    assert torch.equal(rng, torch.get_rng_state())
    torch.manual_seed(73)
    lb = batch_loss(b, batch)["total"]
    lb.backward()
    torch.testing.assert_close(la, lb, atol=0, rtol=0)
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(pa.grad, pb.grad, atol=0, rtol=0)


def test_large_gqa_topology_cpu_correctness_not_benchmark():
    # Worst audited object count, actual D=256/C=311. CPU inference only, no timing claim.
    net = model(311, dim=256).eval()
    batch = collate([graph(126)], 311, torch.Generator().manual_seed(42), negative_ratio=0)
    assert batch["context_pairs"].shape == (2, 15750)
    with torch.no_grad():
        out = net(**model_inputs(batch))
        full_batch = {**batch, "query_pairs": batch["context_pairs"], "query_ptr": batch["context_ptr"]}
        full = net(**model_inputs(full_batch))
    assert out.shape == (1, 311) and torch.isfinite(out).all()
    assert full.shape == (15750, 311) and torch.isfinite(full).all()
    torch.testing.assert_close(out, full[:1], atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize(
    "flags", [{"feedback_enabled": False}, {"reverse_feedback_enabled": False}, {"prototype_update_enabled": False}]
)
def test_ablation_gradient_paths_are_explicit(flags):
    net = model(**flags)
    batch_loss(net, collate([graph()], 51))["total"].backward()
    updates = ("proto_norm", "proto_q", "edge_k", "edge_v", "proto_gru", "update_norm_p")
    for name, parameter in net.feedback.named_parameters():
        if flags.get("feedback_enabled") is False or flags.get("reverse_feedback_enabled") is False:
            assert parameter.grad is None, name
        elif name.startswith(updates[1:]):
            assert parameter.grad is None, name
        else:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
