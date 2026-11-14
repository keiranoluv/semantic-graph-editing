"""Synthetic CPU regressions for a train-only memorization probe."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from test_magicbrush_experiment_v3 import fixture

from tools.emnlp2020_graph_modification_v1.dataset import read, sha256
from tools.magicbrush_overfit_v1 import data, runner


@pytest.fixture(autouse=True)
def cpu_runtime():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def probe_fixture():
    es, store, config, identity = fixture()
    config.update(seed=42, max_steps=100, evaluate_every=10, required_success_checks=2, lr=0.01, weight_decay=0.0)
    config["model"]["dropout"] = 0.0
    config["output"] = "runs/level2/magicbrush-overfit16-v1"
    return es["train"], store, config, identity


def test_selection_is_train_only_balanced_deterministic_and_group_distinct():
    es, _, _, _ = fixture(all_kinds=True)
    pool = [
        replace(es["train"][0], id=f"toy:{i:03}", group=f"group:{i}", instruction=f"Synthetic instruction {i}")
        for i in range(20)
    ]
    a, report = data.select_train(pool, 16, 20261114)
    b, _ = data.select_train(list(reversed(pool)), 16, 20261114)
    assert a == b and len({e.group for e in a}) == 16
    assert len(report["cohort_operation_support"]) == 6
    with pytest.raises(ValueError, match="Only train"):
        data.select_train([*pool, replace(pool[0], split="val")], 16, 20261114)
    conflict = replace(pool[0], id="conflict", target=pool[0].source, operations=())
    selected, report = data.select_train([*pool, conflict], 16, 20261114)
    assert report["conflicting_input_keys_excluded"] == 1
    assert pool[0] not in selected and conflict not in selected


def test_success_requires_raw_and_constrained_full_graph_and_operations():
    perfect = {
        side: {"exact_operation_set_accuracy": 1.0, "exact_target_graph_accuracy": 1.0}
        for side in ["raw", "constrained"]
    }
    assert runner.successful(perfect)
    for side in perfect:
        for key in perfect[side]:
            bad = deepcopy(perfect)
            bad[side][key] = 0.9375
            assert not runner.successful(bad)


@pytest.mark.parametrize("kind", ["mlp", "gnn"])
def test_synthetic_probe_learns_then_stops_and_completed_tampering_rejected(tmp_path, kind):
    cohort, store, config, identity = probe_fixture()
    result = runner.train_one(cohort, store, config, identity, kind, tmp_path, device="cpu")
    assert result["status"] == "MEMORIZED"
    assert result["history"][-1]["success_streak"] == 2
    assert runner.successful(result["evaluation"])
    assert result["evaluation"]["target_conditioned_heads"]["fields"]["node_action"]["positive_accuracy"] == 1
    assert result["evaluation"]["decoded_source_id_and_payload_counts"]["correct_complete_existing_op"] == 1
    assert runner.train_one(cohort, store, config, identity, kind, tmp_path, device="cpu") == result
    (tmp_path / "predictions-train.json").write_text("[]")
    with pytest.raises(ValueError, match="modified"):
        runner.train_one(cohort, store, config, identity, kind, tmp_path, device="cpu")


def test_checkpoint_resume_matches_uninterrupted_training(tmp_path, monkeypatch):
    cohort, store, config, identity = probe_fixture()
    config.update(max_steps=6, evaluate_every=2)
    expected = runner.train_one(cohort, store, config, identity, "gnn", tmp_path / "full", device="cpu")
    real_save = runner.save_checkpoint

    def interrupted(path, payload):
        real_save(path, payload)
        if Path(path).name == "last.pt" and payload["step"] == 2:
            raise KeyboardInterrupt("synthetic interruption after atomic checkpoint")

    monkeypatch.setattr(runner, "save_checkpoint", interrupted)
    with pytest.raises(KeyboardInterrupt):
        runner.train_one(cohort, store, config, identity, "gnn", tmp_path / "resume", device="cpu")
    monkeypatch.setattr(runner, "save_checkpoint", real_save)
    actual = runner.train_one(cohort, store, config, identity, "gnn", tmp_path / "resume", device="cpu")
    assert actual["history"] == expected["history"] and actual["evaluation"] == expected["evaluation"]
    a = torch.load(tmp_path / "full/last.pt", weights_only=True)
    b = torch.load(tmp_path / "resume/last.pt", weights_only=True)
    assert all(torch.equal(v, b["model"][k]) for k, v in a["model"].items())
    assert torch.equal(a["rng"]["cpu"], b["rng"]["cpu"])


def test_both_models_run_despite_failed_budget_and_queue_resumes(tmp_path, monkeypatch):
    cohort, store, config, identity = probe_fixture()
    config.update(max_steps=1, evaluate_every=1, required_success_checks=3)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    calls = []
    interrupt = [True]
    real = runner.train_one

    def train(es, st, cfg, ident, kind, path):
        if kind == "gnn" and interrupt[0]:
            interrupt[0] = False
            raise KeyboardInterrupt("synthetic queue interruption")
        if not (path / "completed-receipt.json").exists():
            calls.append(kind)
        return real(es, st, cfg, ident, kind, path, device="cpu")

    monkeypatch.setattr(runner, "train_one", train)
    with pytest.raises(KeyboardInterrupt):
        runner.execute(config, cohort, store, identity, {})
    runner.execute(config, cohort, store, identity, {})
    out = tmp_path / config["output"]
    summary = read(out / "summary.json")
    assert calls == ["mlp", "gnn"]
    assert all(r["status"] == "NOT_MEMORIZED_WITHIN_BUDGET" for r in summary["models"].values())
    assert summary["automatic_followup_training"] is False
    checksum = sha256(out / "summary.json")
    runner.execute(config, cohort, store, identity, {})
    assert calls == ["mlp", "gnn"] and sha256(out / "summary.json") == checksum
    assert not list(out.glob("**/*test*"))
