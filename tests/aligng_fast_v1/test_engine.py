import json
from dataclasses import asdict

import pytest
import torch

from experiments.aligng_adapted_v1.batching import collate as original_collate
from experiments.aligng_fast_v1 import batching, engine, runner
from experiments.aligng_fast_v1.model import FastConfig
from experiments.aligng_gqa_v1 import engine as original_engine
from tests.aligng_gqa_v1.test_runner import experiment, fixtures


@pytest.mark.parametrize("kind", engine.MODELS)
def test_resume_and_original_engine_isolation(tmp_path, kind):
    study = experiment()
    study["dimensions"]["fast"] = asdict(FastConfig(encoder_chunk_size=7))
    engine.setup_runtime(study["config"])
    train, val, vectors = fixtures()
    train, val = [[batching.prepare_graph(g, 51) for g in gs] for gs in (train, val)]
    full = engine.train_run(kind, 42, train, val, vectors, study, tmp_path / "full")
    for stop in (1, 3):
        folder = tmp_path / f"resume-{stop}"
        assert engine.train_run(kind, 42, train, val, vectors, study, folder, stop_after_batches=stop) is None
        resumed = engine.train_run(kind, 42, train, val, vectors, study, folder)
        for key in ("history", "validation", "initial_state_sha256", "selected_epoch"):
            assert full[key] == resumed[key]
        a, b = [torch.load(p / "last.pt", weights_only=True) for p in (folder, tmp_path / "full")]
        for name in a["model"]:
            torch.testing.assert_close(a["model"][name], b["model"][name], rtol=0, atol=0)
        assert torch.equal(a["sampler_rng"], b["sampler_rng"])
        assert torch.equal(a["torch_rng"], b["torch_rng"])
    assert original_engine.collate is original_collate
    assert original_engine.create_model is not engine.create_model
    assert engine.train_run(kind, 42, train, val, vectors, study, tmp_path / "full") == full


def test_matched_sampling_initialization_and_changed_identity(tmp_path):
    study = experiment()
    study["dimensions"]["fast"] = asdict(FastConfig())
    train, val, vectors = fixtures()
    results = [engine.train_run(k, 42, train, val, vectors, study, tmp_path / k) for k in engine.MODELS]
    assert results[0]["initial_state_sha256"] == results[1]["initial_state_sha256"]
    assert [r["sampling_trace_sha256"] for r in results[0]["history"]] == [
        r["sampling_trace_sha256"] for r in results[1]["history"]
    ]
    study["dimensions"]["fast"]["encoder_chunk_size"] = 2
    with pytest.raises(ValueError, match="identity"):
        engine.train_run(engine.MODELS[0], 42, train, val, vectors, study, tmp_path / engine.MODELS[0])


def test_online_wandb_required_before_full_training(tmp_path):
    study = experiment()
    study["config"]["scope"] = "full"
    study["dimensions"]["fast"] = asdict(FastConfig())
    train, val, vectors = fixtures()
    with pytest.raises(RuntimeError, match="online W&B"):
        engine.train_run(engine.MODELS[0], 42, train, val, vectors, study, tmp_path)
    assert not (tmp_path / "last.pt").exists()


def test_isolated_config_and_matrix(tmp_path):
    cfg = runner.load_config()
    assert cfg["output"] == "runs/level2/aligng-gqa-fast-v1/full"
    assert cfg["train_batch_size"] == cfg["val_batch_size"] == 512
    assert runner.queue() == [(m, s) for s in (42, 123, 3407) for m in engine.MODELS]
    cfg["output"] = "runs/level2/aligng-gqa-bs512-v1/full"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):
        runner.load_config(path)


def test_probe_uses_new_factory_and_matches_evaluation():
    from experiments.aligng_fast_v1.probe import measure

    study = experiment()
    study["dimensions"]["fast"] = asdict(FastConfig())
    _, graphs, vectors = fixtures()
    results = [
        measure(engine.MODELS[0], graphs, False, study, vectors, device="cpu", warmup=0, repeats=1, variant=v)
        for v in ("original", "fast")
    ]
    torch.testing.assert_close(results[0]["_logits"], results[1]["_logits"], rtol=1e-4, atol=2e-4)
    result = measure(engine.MODELS[0], graphs, True, study, vectors, device="cpu", warmup=1, repeats=1)
    assert result["status"] == "PASS" and result["measurements"][0]["gradient_norm"] > 0
