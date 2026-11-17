import copy
import json
from dataclasses import asdict

import pytest
import torch

from experiments.aligng_ablation_v1 import engine, runner
from experiments.aligng_ablation_v1.probe import measure
from experiments.aligng_adapted_v1 import AlignGConfig, AlignGPredicateClassifier
from experiments.aligng_adapted_v1 import batching as slow_batching
from experiments.aligng_fast_v1 import batching
from experiments.aligng_fast_v1.model import FastConfig
from tests.aligng_gqa_v1.test_runner import experiment, fixtures


def study():
    exp = experiment()
    exp["dimensions"]["fast"] = asdict(FastConfig(encoder_chunk_size=7))
    return exp


@pytest.mark.parametrize("kind", engine.MODELS)
def test_resume_loss_dispatch_and_completion(tmp_path, kind):
    exp = study()
    engine.setup_runtime(exp["config"])
    train, val, vectors = fixtures()
    train, val = [[batching.prepare_graph(g, 51) for g in gs] for gs in (train, val)]
    full = engine.train_run(kind, 42, train, val, vectors, exp, tmp_path / "full")
    for stop in (1, 3):
        folder = tmp_path / f"resume-{stop}"
        assert engine.train_run(kind, 42, train, val, vectors, exp, folder, stop_after_batches=stop) is None
        resumed = engine.train_run(kind, 42, train, val, vectors, exp, folder)
        for key in ("history", "validation", "initial_state_sha256", "selected_epoch"):
            assert full[key] == resumed[key]
        a, b = [torch.load(p / "last.pt", weights_only=True) for p in (folder, tmp_path / "full")]
        for name in a["model"]:
            torch.testing.assert_close(a["model"][name], b["model"][name], rtol=0, atol=0)
    if kind == "pe-ce-aligng":
        for row in full["history"]:
            loss = row["train_losses"]
            assert loss["pl_euclidean"] == loss["pr_cosine"] == loss["pr_euclidean"] == 0
            assert loss["total"] == loss["ce"]
    else:
        assert full["history"][0]["train_losses"]["pr_euclidean"] > 0
    assert engine.train_run(kind, 42, train, val, vectors, exp, tmp_path / "full") == full
    changed = copy.deepcopy(exp)
    changed["dimensions"]["fast"]["encoder_chunk_size"] = 3
    with pytest.raises(ValueError, match="identity|changed"):
        engine.train_run(kind, 42, train, val, vectors, changed, tmp_path / "full")


@pytest.mark.parametrize("kind", engine.MODELS)
def test_fast_matches_slow_ablation_and_gradient_path(kind):
    exp = study()
    _, graphs, vectors = fixtures()
    torch.manual_seed(42)
    fast = engine.create_model(kind, exp["dimensions"], vectors)
    slow = AlignGPredicateClassifier(
        *vectors,
        semantic_dim=16,
        dropout=0.2,
        config=AlignGConfig(**{**exp["dimensions"]["aligng"], **engine.VARIANTS[kind]}),
    )
    slow.load_state_dict(fast.state_dict())
    fast.eval()
    slow.eval()
    fb = batching.collate(graphs, 51)
    sb = slow_batching.collate(graphs, 51)
    a = fast(**batching.model_inputs(fb, "cpu"))
    b = slow(**slow_batching.model_inputs(sb))
    torch.testing.assert_close(a, b, rtol=1e-4, atol=2e-4)
    a.square().sum().backward()
    named = dict(fast.named_parameters())
    gru = [p for n, p in named.items() if "gru" in n.lower()]
    assert gru
    if kind == "edge-only":
        assert all(p.grad is None for p in gru)
    else:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in gru)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in named.items() if "edge_ffn" in n)


def test_full_requires_online_wandb(tmp_path):
    exp = study()
    exp["config"]["scope"] = "full"
    train, val, vectors = fixtures()
    with pytest.raises(RuntimeError, match="online W&B"):
        engine.train_run(engine.MODELS[0], 42, train, val, vectors, exp, tmp_path)
    assert not (tmp_path / "last.pt").exists()


def test_exact_queue_and_recipe(tmp_path):
    assert runner.queue() == [(m, s) for s in (42, 123, 3407) for m in ("edge-only", "pe-ce-aligng")]
    cfg = runner.load_config()
    cfg["train_batch_size"] = 16
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="preserve"):
        runner.load_config(path)


@pytest.mark.parametrize("kind", engine.MODELS)
def test_probe_cpu_forward_backward(kind):
    exp = study()
    _, graphs, vectors = fixtures()
    measured = measure(kind, graphs, True, exp, vectors, device="cpu", warmup=0, repeats=1)
    assert measured["status"] == "PASS"
    assert measured["measurements"][0]["gradient_norm"] > 0
