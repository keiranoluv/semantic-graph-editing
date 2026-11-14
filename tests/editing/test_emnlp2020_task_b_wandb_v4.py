"""CPU protocol tests and W&B fakes: no real W&B initialization or GPU workload."""

import copy
import math
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from src.editing.operations import EditGraph, Node, Operation, apply_operations
from tools.edit_model_v1.data import Example
from tools.edit_model_v1.text import TextStore, text_key
from tools.emnlp2020_graph_modification_v1.dataset import read, write
from tools.emnlp2020_task_b_gpu_v3.runner import inputs_tensor
from tools.emnlp2020_task_b_v2.model import ActionVocabulary
from tools.emnlp2020_task_b_wandb_v4 import runner, tracking, training


@pytest.fixture
def fixture(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(314)
    source = EditGraph((Node("n001", "cat"),), ())
    ops = (Operation("replace_node", source.nodes[0], Node("n001", "dog")),)
    e = Example("train", "train", "a", "make dog", source, apply_operations(source, ops), ops)
    rows = {"train": [e], "val": [replace(e, id="val", split="val", group="b")]}
    store = TextStore([text_key(t) for t in [e.instruction, '["cat", []]']], torch.randn(2, 512))
    config = read("configs/level2/emnlp2020-task-b-wandb-v4.json")
    config.update(
        device="cpu",
        execution="smoke",
        output=str(tmp_path / "run"),
        dataset=str(tmp_path / "data"),
        epochs=3,
        minimum_epochs=1,
        patience=10,
        models=["mlp", "gnn"],
        seeds=[42, 43, 44],
        hidden=8,
        max_nodes=1,
        max_edges=1,
        max_decode_tokens=12,
        batch_size=2,
    )
    write(tmp_path / "data/trainval-receipt.json", {"fixture": True})
    write(tmp_path / "data/clip-trainval/manifest.json", {"fixture": True})
    vocabulary = ActionVocabulary.fit(rows["train"])
    tensors = {k: inputs_tensor(es, store, config) for k, es in rows.items()}
    return config, rows, vocabulary, store, tensors


class FakeRun:
    def __init__(self):
        self.id = "stable-id"
        self.url = "https://example.invalid/run/stable-id"
        self.settings = SimpleNamespace(mode="online")
        self.step = 0
        self.logs = []
        self.summary = {}
        self.finishes = []
        self.fail_log_once = False

    def define_metric(self, *args, **kwargs):
        pass

    def log(self, row, step):
        self.logs.append((step, row))
        self.step = step + 1
        if self.fail_log_once:
            self.fail_log_once = False
            raise RuntimeError("simulated tracking disconnect after durable checkpoint")

    def finish(self, exit_code):
        self.finishes.append(exit_code)


def test_early_stop_minimum_patience_delta_and_independent_best_selection():
    config = {"minimum_epochs": 6, "patience": 3, "min_delta": 0.001}
    state = training.initial_state()
    for loss in [1.0, 0.9998, 0.9996, 0.9994, 0.9992]:
        state, selected = training.advance(state, loss, config)
        assert selected
        assert not training.should_stop(state, config)
    state, selected = training.advance(state, 0.9991, config)
    assert selected and state["best_epoch"] == 6
    assert state["bad_epochs"] == 5
    assert training.should_stop(state, config)
    with pytest.raises(ValueError, match="nonfinite"):
        training.advance(state, math.nan, config)


def test_online_tracking_arguments_stable_ids_and_no_raw_records(fixture, monkeypatch):
    config, _, _, _, _ = fixture
    config["execution"] = "main"
    calls = []
    fake = FakeRun()
    monkeypatch.delenv("WANDB_MODE", raising=False)

    def init(**kwargs):
        calls.append(kwargs)
        return fake

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init, Settings=lambda **kw: kw))
    ident = {"experiment_identity": "abc", "model": "gnn", "seed": 42}
    assert tracking.start(config, ident) is fake
    tracking.start(config, ident)
    assert calls[0]["id"] == calls[1]["id"]
    assert calls[0]["mode"] == "online" and calls[0]["force"] is True
    assert calls[0]["resume"] == "allow" and calls[0]["save_code"] is False
    assert "vocabulary" not in calls[0]["config"]
    assert "instruction" not in calls[0]["config"]
    config["execution"] = "smoke"
    assert tracking.start(config, ident) is None
    assert len(calls) == 2


def test_main_rejects_offline_or_missing_tracker_before_gradients(fixture, monkeypatch):
    config, rows, vocabulary, _, tensors = fixture
    config["execution"] = "main"
    monkeypatch.setenv("WANDB_MODE", "offline")
    with pytest.raises(ValueError, match="WANDB_MODE"):
        training.train_one(config, "mlp", 42, rows, tensors, vocabulary, "identity")
    assert not (runner.Path(config["output"]) / "mlp-seed42/last.pt").exists()
    monkeypatch.setenv("WANDB_MODE", "online")
    fake = FakeRun()
    fake.settings.mode = "offline"
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=lambda **kw: fake, Settings=lambda **kw: kw))
    with pytest.raises(RuntimeError, match="not_initialized"):
        tracking.start(config, {"experiment_identity": "x", "model": "mlp", "seed": 42})
    assert fake.finishes == [1]


@pytest.mark.parametrize("kind", ["mlp", "gnn"])
def test_resume_restores_optimizer_rng_best_epoch_and_online_metrics(fixture, monkeypatch, kind):
    config, rows, vocabulary, _, tensors = fixture
    config["execution"] = "main"  # Direct CPU contract exercise; CLI main requires CUDA.
    resumed = FakeRun()
    resumed.fail_log_once = True
    monkeypatch.setattr(tracking, "start", lambda *args: resumed)
    with pytest.raises(RuntimeError, match="disconnect"):
        training.train_one(config, kind, 42, rows, tensors, vocabulary, "same")
    path = runner.Path(config["output"]) / f"{kind}-seed42"
    saved = torch.load(path / "last.pt", weights_only=True)
    assert saved["stop_state"]["epoch"] == 1
    assert not (path / "result.json").exists() and resumed.finishes == [1]
    completed = training.train_one(config, kind, 42, rows, tensors, vocabulary, "same")
    assert resumed.finishes[-1] == 0
    assert [step for step, row in resumed.logs] == [1, 2, 3]
    assert "val/per_type/replace_node/f1" in resumed.logs[-1][1]
    assert "selected_val/micro/f1" in resumed.summary
    assert completed["wandb"]["mode"] == "online"
    other = copy.deepcopy(config)
    other["output"] += "-uninterrupted"
    fresh = FakeRun()
    monkeypatch.setattr(tracking, "start", lambda *args: fresh)
    uninterrupted = training.train_one(other, kind, 42, rows, tensors, vocabulary, "same")
    assert completed["history"] == uninterrupted["history"]
    actual = torch.load(path / "best.pt", weights_only=True)
    expected = torch.load(runner.Path(other["output"]) / f"{kind}-seed42/best.pt", weights_only=True)
    assert all(torch.equal(actual["state_dict"][k], expected["state_dict"][k]) for k in actual["state_dict"])
    assert completed["best_epoch"] == uninterrupted["best_epoch"]
    monkeypatch.setattr(tracking, "start", lambda *args: pytest.fail("completed runs must be skipped"))
    assert training.train_one(config, kind, 42, rows, tensors, vocabulary, "same") == completed
    (path / "best.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="artifact_changed"):
        training.train_one(config, kind, 42, rows, tensors, vocabulary, "same")


def test_smoke_queue_uses_no_wandb_or_test_and_resumes_after_report_commit(fixture, monkeypatch):
    config, rows, vocabulary, store, tensors = fixture
    config.update(epochs=1, minimum_epochs=1, patience=1)
    monkeypatch.setattr(runner, "prepare_inputs", lambda *args: (rows, vocabulary, store, tensors))
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=lambda **kw: pytest.fail("W&B must stay untouched")))
    freeze = runner.train(config)
    assert len(freeze["runs"]) == 6 and freeze["test_used"] is False
    assert freeze["all_runs_tracked_online"] is False and freeze["automatic_test"] is False
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **kw: "different_report_commit")
    assert runner.train(config) == freeze
    changed = copy.deepcopy(config)
    changed["patience"] = 2
    with pytest.raises(ValueError, match="use_new_version"):
        runner.train(changed)


def test_config_stops_missing_wandb_and_invalid_stopping_parameters(fixture):
    config, *_ = fixture
    for key, value in [("patience", 0), ("minimum_epochs", 100), ("min_delta", math.nan), ("lr", 0)]:
        bad = {**config, key: value}
        with pytest.raises(ValueError):
            runner.validate_config(bad)
    config.update(execution="main", device="cuda", wandb={"mode": "disabled"})
    with pytest.raises(ValueError, match="online_WandB"):
        runner.validate_config(config)
