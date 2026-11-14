import copy
import json
from dataclasses import asdict

import pytest
import torch

from experiments.aligng_adapted_v1 import AlignGConfig
from experiments.aligng_gqa_v1 import engine, runner
from src.datasets.gqa_full import digest
from tests.aligng_adapted_v1.conftest import graph


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(1)


def experiment():
    cfg = runner.load_config(runner.DEFAULT)
    cfg.update(device="cpu", epochs=2, train_batch_size=2, val_batch_size=2, checkpoint_every_batches=2)
    return {
        "config": cfg,
        "dimensions": {
            "num_classes": 51,
            "semantic_dim": 16,
            "pe_dropout": 0.2,
            "aligng": asdict(AlignGConfig(pair_chunk_size=3, attention_query_chunk_size=3)),
        },
    }


def fixtures():
    torch.manual_seed(42)
    return (
        [graph(3), graph(0, ()), graph(1, ()), graph(4, seed=37), graph(3, seed=31)],
        [graph(3, seed=71)],
        [torch.randn(8, 12), torch.randn(51, 12)],
    )


@pytest.mark.parametrize("kind", engine.MODELS)
def test_resume_between_batches_and_epoch_boundary_matches_exactly(tmp_path, kind):
    study = experiment()
    engine.setup_runtime(study["config"])
    train, val, vectors = fixtures()
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
    # A complete run verifies and returns without another optimizer step.
    assert engine.train_run(kind, 42, train, val, vectors, study, tmp_path / "full") == full


def test_paired_sampling_initialization_summary_and_checkpoint_tampering(tmp_path):
    study = experiment()
    train, val, vectors = fixtures()
    for kind in engine.MODELS:
        engine.train_run(kind, 42, train, val, vectors, study, tmp_path / kind / "42")
    summary = runner.summarize(study, tmp_path)
    assert summary["matched_sampling"] and summary["matched_initial_state"]
    checkpoint = tmp_path / engine.MODELS[0] / "42/best.pt"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checkpoint"):
        runner.summarize(study, tmp_path)


def test_changed_config_cannot_resume(tmp_path):
    study = experiment()
    train, val, vectors = fixtures()
    engine.train_run(engine.MODELS[0], 42, train, val, vectors, study, tmp_path, stop_after_batches=1)
    study["config"]["lr"] *= 2
    with pytest.raises(ValueError, match="identity"):
        engine.train_run(engine.MODELS[0], 42, train, val, vectors, study, tmp_path)


@pytest.mark.parametrize("change", [{"scope": "full"}, {"epochs": 6}, {"train_batch_size": 0}, {"lr": float("nan")}])
def test_config_rejects_untracked_full_and_invalid_pilot(tmp_path, change):
    cfg = runner.load_config(runner.DEFAULT)
    cfg.update(change)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):
        runner.load_config(path)


def test_single_job_cli_requires_explicit_model_seed_before_loading(monkeypatch):
    def forbidden(*args):
        raise AssertionError("Must not load assets for incomplete CLI")

    monkeypatch.setattr(runner, "prepare", forbidden)
    with pytest.raises(SystemExit):
        runner.main(["--stage", "pilot"])


def test_full_run_fails_closed_if_tracker_disabled(tmp_path, monkeypatch):
    study = experiment()
    study["config"]["scope"] = "full"
    train, val, vectors = fixtures()
    with pytest.raises(RuntimeError, match="online W&B"):
        engine.train_run(engine.MODELS[0], 42, train, val, vectors, study, tmp_path)
    assert not (tmp_path / "last.pt").exists()


def test_tracker_receives_durable_history_and_full_settings(tmp_path, monkeypatch):
    study = experiment()
    study["config"].update(scope="full", wandb={"mode": "online"})
    calls = []

    class Tracker:
        mode = "online"
        run = type("Run", (), {"settings": type("Settings", (), {"mode": "online"})()})()

        def __init__(self, exp, identity, directory):
            assert identity["experiment_sha256"] == digest(exp)
            self.directory = directory

        def replay(self, rows):
            if rows:
                saved = torch.load(self.directory / "last.pt", weights_only=True)
                assert saved["history"][-1]["epoch"] >= rows[-1]["epoch"]
            calls.append(copy.deepcopy(rows))

        def complete(self, result):
            assert (self.directory / "result.json").exists()

        def finish(self, status):
            assert status == "completed"

    monkeypatch.setattr(engine, "RunTracker", Tracker)
    train, val, vectors = fixtures()
    result = engine.train_run(engine.MODELS[0], 42, train, val, vectors, study, tmp_path)
    assert len(result["history"]) == 2 and len(calls) == 3
