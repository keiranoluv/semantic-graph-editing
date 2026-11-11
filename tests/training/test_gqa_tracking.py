"""Early stopping/checkpoint and tracking tests, using synthetic fixtures only."""

import copy
import json
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.datasets.gqa_full import atomic_write, digest
from src.training.gqa_full import summarize_matrix, train_run, update_early_stopping
from src.training.gqa_tracking import RunTracker
from tests.training.test_gqa_full import fixtures
from tools.run_gqa_full import load_config


def test_v2_config_requested_budget_and_validation(tmp_path):
    config = load_config("configs/level2/gqa-full-v2.json")
    assert config["epochs"] == 200
    assert config["early_stopping"] == {"enabled": True, "patience": 15, "min_delta": 0.0}
    assert config["wandb"] == {"mode": "online", "entity": "keiranoluv-hcmus", "project": "semantic-graph-editing"}
    config["early_stopping"]["patience"] = 0
    atomic_write(tmp_path / "bad.json", config)
    with pytest.raises(ValueError, match="early stopping"):
        load_config(tmp_path / "bad.json")


def test_patience_fifteen_ties_reset_and_min_delta():
    state = {"bad_epochs": 0, "stopping_best": -1.0, "stopped_early": False}
    settings = {"enabled": True, "patience": 15, "min_delta": 0.0}
    update_early_stopping(state, 0.5, settings)
    for n in range(1, 16):
        update_early_stopping(state, 0.5, settings)
        assert state["bad_epochs"] == n
        assert state["stopped_early"] == (n == 15)
    update_early_stopping(state, 0.6, settings)
    assert state["bad_epochs"] == 0 and not state["stopped_early"]
    settings["min_delta"] = 0.05
    update_early_stopping(state, 0.61, settings)
    assert state["bad_epochs"] == 1 and state["stopping_best"] == 0.6
    update_early_stopping(state, 0.66, settings)
    assert state["bad_epochs"] == 0


def fixed_evaluation(*args, **kwargs):
    return {"loss_all_pairs": 1.0, "gc": {k: 0.5 for k in ("R@50", "R@100", "mR@50", "mR@100")}, "ng": {"ng-R@50": 0.5}}


def test_early_stop_resume_preserves_patience_and_best_before_finish(tmp_path, monkeypatch):
    from src.training import gqa_full

    monkeypatch.setattr(gqa_full, "evaluate", fixed_evaluation)
    torch.set_num_threads(1)
    train, val, vectors, experiment = fixtures("cpu")
    experiment["config"].update(epochs=20, early_stopping={"enabled": True, "patience": 2, "min_delta": 0.0})
    expected = train_run("pe-plpr", 42, train, val, vectors, experiment, tmp_path / "full")
    partial = tmp_path / "partial"
    assert train_run("pe-plpr", 42, train, val, vectors, experiment, partial, stop_after_batches=7) is None
    saved = torch.load(partial / "last.pt", weights_only=True)
    best = torch.load(partial / "best.pt", weights_only=True)
    assert saved["bad_epochs"] == 1 and saved["epoch"] == 3
    assert best["epoch"] == 1
    assert not (partial / "result.json").exists()
    (partial / "best.pt").unlink()  # rebuild the derived export from authoritative last.pt
    actual = train_run("pe-plpr", 42, train, val, vectors, experiment, partial)
    assert actual["history"] == expected["history"]
    assert actual["epochs_completed"] == 3 and actual["stopping_reason"] == "early_stopping"
    assert actual["selected_epoch"] == 1 and actual["early_stopping_bad_epochs"] == 2
    best = torch.load(partial / "best.pt", weights_only=True)
    last = torch.load(partial / "last.pt", weights_only=True)
    assert any(not torch.equal(best["model"][k], last["model"][k]) for k in best["model"])
    assert last["stopped_early"] and last["epoch"] == 4


def test_cloud_failure_after_epoch_keeps_committed_training(tmp_path, monkeypatch):
    train, val, vectors, experiment = fixtures("cpu")
    original = RunTracker.replay

    def fail(self, history):
        if history:
            raise RuntimeError("simulated logging failure")

    monkeypatch.setattr(RunTracker, "replay", fail)
    with pytest.raises(RuntimeError, match="logging failure"):
        train_run("pe-plpr", 42, train, val, vectors, experiment, tmp_path)
    last = torch.load(tmp_path / "last.pt", weights_only=True)
    assert last["epoch"] == 2 and len(last["history"]) == 1
    monkeypatch.setattr(RunTracker, "replay", original)
    assert train_run("pe-plpr", 42, train, val, vectors, experiment, tmp_path)["epochs_completed"] == 2


def test_best_is_absolute_max_even_when_gain_below_stopping_delta(tmp_path, monkeypatch):
    from src.training import gqa_full

    scores = iter([0.5, 0.51, 0.52, 0.52])  # three epochs, then selected-checkpoint evaluation

    def evaluate(*args, **kwargs):
        value = fixed_evaluation()
        value["gc"]["mR@50"] = next(scores)
        return value

    monkeypatch.setattr(gqa_full, "evaluate", evaluate)
    train, val, vectors, experiment = fixtures("cpu")
    experiment["config"].update(epochs=20, early_stopping={"enabled": True, "patience": 2, "min_delta": 0.1})
    result = train_run("pe-plpr", 42, train, val, vectors, experiment, tmp_path)
    assert result["epochs_completed"] == 3 and result["stopping_reason"] == "early_stopping"
    assert result["selected_epoch"] == 3
    assert torch.load(tmp_path / "best.pt", weights_only=True)["score"] == 0.52


class FakeRun:
    def __init__(self, step=0):
        self.step, self.summary, self.events = step, {}, []
        self.dir, self.url = "fixture/files", "https://example.invalid/fixture"

    def define_metric(self, *args, **kwargs):
        pass

    def log(self, values, step, commit):
        assert commit and step >= self.step
        self.events.append(copy.deepcopy(values))
        self.step = step + 1
        torch.rand(2)  # even an SDK that consumes RNG cannot perturb training
        np.random.rand()
        random.random()

    def finish(self, exit_code):
        self.exit_code = exit_code


def epoch_row(epoch):
    return {
        "epoch": epoch,
        "train_losses": {"ce": 1.0},
        "validation": {"mR@50": 0.1},
        "train_pairs": 10,
        "steps": 1,
        "max_gradient_norm_before_clip": 2.0,
        "best_score": 0.1,
        "best_epoch": 1,
        "early_stopping_bad_epochs": epoch - 1,
    }


@pytest.mark.parametrize("mode", ["online", "offline"])
def test_tracking_stable_online_id_offline_snapshots_replay_and_rng(tmp_path, monkeypatch, mode):
    created, requests, server_step = [], [], {"value": 0}

    def init(**kwargs):
        requests.append(kwargs)
        run = FakeRun(server_step["value"] if mode == "online" else 0)
        created.append(run)
        return run

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init, Settings=lambda **kwargs: kwargs))
    experiment = {"config": {"scope": "fixture", "wandb": {"mode": mode, "project": "fixture", "entity": "fixture"}}}
    identity = {"model": "pe-plpr", "seed": 42, "experiment_sha256": "fixture"}
    tracker = RunTracker(experiment, identity, tmp_path)
    rng = torch.get_rng_state()
    tracker.replay([epoch_row(1)])
    tracker.replay([epoch_row(1)])
    assert len(created[-1].events) == 1
    assert torch.equal(rng, torch.get_rng_state())
    server_step["value"] = created[-1].step
    tracker.finish("interrupted")
    assert created[-1].exit_code == 130
    tracker = RunTracker(experiment, identity, tmp_path)
    tracker.replay([epoch_row(1), epoch_row(2)])
    assert [r["epoch"] for r in created[-1].events] == ([2] if mode == "online" else [1, 2])
    assert (requests[0]["id"] == requests[1]["id"]) == (mode == "online")
    assert requests[0]["resume"] == ("allow" if mode == "online" else None)
    tracker.finish("completed")
    assert json.loads((tmp_path / "wandb-latest.json").read_text())["execution_status"] == "completed"


def test_summary_common_prefix_accepts_different_stop_epochs_rejects_drift(tmp_path):
    from src.knowledge.transe.core import sha256

    experiment = {"config": {"scope": "fixture", "models": ["pe-ce", "pe-pl"], "seeds": [42]}}
    for model, epochs in [("pe-ce", 2), ("pe-pl", 3)]:
        path = tmp_path / model / "42"
        path.mkdir(parents=True)
        (path / "best.pt").write_bytes(b"fixture")
        r = {
            "status": "PASS",
            "identity": {"experiment_sha256": digest(experiment), "model": model, "seed": 42},
            "checkpoint_sha256": sha256(path / "best.pt"),
            "history": [{"sampling_trace_sha256": str(i)} for i in range(epochs)],
            "initial_state_sha256": "same",
            "epochs_completed": epochs,
            "stopping_reason": "early_stopping",
            "selected_epoch": 1,
            "validation": {
                protocol: {prefix + k: 0.1 for k in ["R@50", "R@100", "mR@50", "mR@100"]}
                for protocol, prefix in [("gc", ""), ("ng", "ng-")]
            },
        }
        atomic_write(path / "result.json", r)
    assert summarize_matrix(experiment, tmp_path)["matched_sampling"]
    path = tmp_path / "pe-pl/42/result.json"
    r = json.loads(path.read_text())
    r["history"][0]["sampling_trace_sha256"] = "wrong"
    atomic_write(path, r)
    with pytest.raises(AssertionError, match="Sampling"):
        summarize_matrix(experiment, tmp_path)
