import copy
from dataclasses import replace

import pytest
import torch

from experiments.aligng_adapted_v1 import batching as original
from experiments.aligng_fast_v1 import batching as fast
from experiments.aligng_fast_v1.model import FastAlignG, FastConfig
from tests.aligng_adapted_v1.conftest import graph, model


@pytest.fixture(autouse=True)
def cpu():
    torch.set_num_threads(1)
    torch.manual_seed(42)


def models(classes=51, dropout=0, **flags):
    reference = model(classes, dropout=dropout, **flags)
    optimized = FastAlignG(
        reference.object_embedding.weight.detach(),
        reference.predicate_embedding.weight.detach(),
        semantic_dim=16,
        dropout=dropout,
        config=reference.config,
        fast_config=FastConfig(encoder_chunk_size=7),
    )
    optimized.load_state_dict(reference.state_dict(), strict=True)
    return reference, optimized


def examples():
    a = graph(5)
    a.edge_index = a.edge_index[:, torch.randperm(a.edge_index.shape[1])]
    a.edge_label_index = a.edge_label_index[:, torch.randperm(a.edge_label_index.shape[1])]
    return [graph(0, ()), a, graph(1, ()), graph(3, ()), graph(4, seed=3)]


@pytest.mark.parametrize("sampled", [True, False])
def test_same_sampling_targets_order_and_rng(sampled):
    graphs = examples()
    a, b = [torch.Generator().manual_seed(123) for _ in range(2)]
    old = original.collate(graphs, 51, a if sampled else None)
    new = fast.collate([fast.prepare_graph(g, 51) for g in graphs], 51, b if sampled else None)
    for key in old:
        torch.testing.assert_close(old[key], new[key], rtol=0, atol=0)
    assert torch.equal(a.get_state(), b.get_state())
    assert torch.equal(new["context_pairs"][:, new["query_context"]], new["query_pairs"])
    inputs = fast.model_inputs(new)["batch"]
    assert "targets" not in inputs.__dataclass_fields__


@pytest.mark.parametrize("classes", [51, 311])
@pytest.mark.parametrize(
    "flags", [{}, {"prototype_update_enabled": False}, {"feedback_enabled": False}, {"reverse_feedback_enabled": False}]
)
def test_reference_logits_embeddings_and_all_gradients(classes, flags):
    old, new = models(classes, **flags)
    graphs = examples()
    graphs[1].x.requires_grad_()
    ob, nb = original.collate(graphs, classes), fast.collate(graphs, classes)
    a = old(**original.model_inputs(ob), return_embeddings=True)
    b = new(**fast.model_inputs(nb), return_embeddings=True)
    for left, right in zip(a, b, strict=True):
        torch.testing.assert_close(left, right, rtol=2e-5, atol=3e-6)
    sum(t.square().mean() for t in a).backward()
    sum(t.square().mean() for t in b).backward()
    for name, p in old.named_parameters():
        q = dict(new.named_parameters())[name]
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, rtol=3e-4, atol=3e-5, msg=name)
    assert graphs[1].x.grad is None


def test_padding_image_isolation_query_order_and_empty():
    _, m = models()
    m.eval()
    graphs = examples()
    batch = fast.collate(graphs, 51)
    with torch.no_grad():
        combined = m(**fast.model_inputs(batch))
        separate = torch.cat([m(**fast.model_inputs(fast.collate([g], 51))) for g in graphs])
        torch.testing.assert_close(combined, separate, rtol=2e-5, atol=3e-6)
        reverse = m(**fast.model_inputs(fast.collate(graphs[::-1], 51)))
        lengths = [g.edge_index.shape[1] for g in graphs]
        torch.testing.assert_close(combined, torch.cat(list(reverse.split(lengths[::-1]))[::-1]), rtol=2e-5, atol=3e-6)
        changed = copy.deepcopy(graphs)
        changed[-1].x *= 300
        result = m(**fast.model_inputs(fast.collate(changed, 51)))
        torch.testing.assert_close(combined[: -lengths[-1]], result[: -lengths[-1]], rtol=0, atol=0)


def test_labels_and_query_sampling_cannot_change_adapted_prototypes():
    _, m = models(dropout=0.2)
    graphs = [graph(5), graph(3), graph(4)]
    other = copy.deepcopy(graphs)
    other[0].gt_predicate[:] = 4
    captured = []
    hook = m.feedback.proto_gru.register_forward_hook(lambda _m, _i, o: captured.append(o.detach().clone()))
    for gs, sampled in ((graphs, False), (other, True)):
        torch.manual_seed(91)
        m(**fast.model_inputs(fast.collate(gs, 51, torch.Generator().manual_seed(9) if sampled else None)))
    hook.remove()
    n = len(captured) // 2
    assert n > 0
    for a, b in zip(captured[:n], captured[n:], strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("encoder,attention", [(False, False), (False, True), (True, True)])
def test_checkpoint_options_preserve_dropout_outputs_gradients_rng(encoder, attention):
    _, a = models(dropout=0.2)
    b = copy.deepcopy(a)
    b.fast_config = replace(a.fast_config, checkpoint_encoder=encoder, checkpoint_attention=attention)
    batch = fast.model_inputs(fast.collate(examples(), 51))
    outputs, states = [], []
    for m in (a, b):
        torch.manual_seed(7)
        logits = m(**batch)
        logits.square().mean().backward()
        outputs.append(logits)
        states.append(torch.get_rng_state())
    torch.testing.assert_close(*outputs, rtol=0, atol=0)
    assert torch.equal(*states)
    for p, q in zip(a.parameters(), b.parameters(), strict=True):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["topology", "object", "feature", "predicate"])
def test_invalid_input_is_rejected_on_cpu(kind):
    g = graph()
    if kind == "topology":
        g.edge_index = g.edge_index[:, :-1]
    elif kind == "object":
        g.object_labels[0] = 1700
    elif kind == "feature":
        g.x[0, 0] = float("nan")
    else:
        g.gt_predicate[0] = 51
    with pytest.raises(ValueError):
        fast.prepare_graph(g, 51)


def test_forward_has_no_scalar_extraction_or_sorting():
    _, m = models()
    m.eval()
    batch = fast.model_inputs(fast.collate(examples(), 51))
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as p:
        m(**batch)
    names = {event.key for event in p.key_averages()}
    assert not names & {"aten::item", "aten::_local_scalar_dense", "aten::sort", "aten::searchsorted"}
