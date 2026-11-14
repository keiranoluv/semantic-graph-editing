import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from experiments.aligng_fast_v1 import probe, runner
from experiments.aligng_fast_v1.model import FastConfig
from src.datasets.gqa_full import atomic_write, digest
from src.knowledge.transe.core import sha256
from tests.aligng_gqa_v1.test_runner import experiment, fixtures


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(1)


def study(tmp_path):
    exp = experiment()
    exp["dimensions"]["fast"] = asdict(FastConfig())
    exp["config"].update(output=str(tmp_path), models=list(runner.MODELS), seeds=list(runner.SEEDS))
    exp["queue_order"] = [{"model": m, "seed": s} for m, s in runner.queue()]
    return exp


def test_recipe_all_batches512_and_online():
    cfg = runner.load_config()
    assert cfg["train_batch_size"] == cfg["val_batch_size"] == 512
    assert cfg["checkpoint_every_batches"] == 10
    assert cfg["wandb"]["mode"] == "online"
    assert runner.queue() == [
        ("pe-plpr-aligng", 42),
        ("pe-plpr", 42),
        ("pe-plpr-aligng", 123),
        ("pe-plpr", 123),
        ("pe-plpr-aligng", 3407),
        ("pe-plpr", 3407),
    ]


def test_real_six_run_cpu_matrix_summary_and_completed_skip(tmp_path, monkeypatch):
    exp = study(tmp_path)
    train, val, vectors = fixtures()
    assert runner.run_queue(exp, train, val, vectors) == 0
    state = json.loads((tmp_path / "queue-status.json").read_text())
    assert state["status"] == "completed" and len(state["completed"]) == 6
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["matched_sampling"] and summary["matched_pe_initialization"]
    assert summary["seeds"] == [42, 123, 3407]

    def forbidden(*args, **kwargs):
        raise AssertionError("Completed jobs must be verified and skipped before W&B/optimizer")

    monkeypatch.setattr(runner.engine, "train_run", forbidden)
    assert runner.run_queue(exp, train, val, vectors) == 0
    checkpoint = tmp_path / "pe-plpr-aligng/42/best.pt"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"mutation")
    with pytest.raises(ValueError, match="checkpoint"):
        runner.run_queue(exp, train, val, vectors)


@pytest.mark.parametrize("outcome", ["interrupt", "error", "missing-result"])
def test_queue_stops_without_launching_next_job(tmp_path, monkeypatch, outcome):
    exp = study(tmp_path)
    calls = []

    def fake(*args, **kwargs):
        calls.append(args[:2])
        if outcome == "interrupt":
            return None
        if outcome == "error":
            raise RuntimeError("test failure")
        return {"status": "PASS"}

    monkeypatch.setattr(runner.engine, "train_run", fake)
    if outcome == "interrupt":
        assert runner.run_queue(exp, [], [], []) == 130
    else:
        with pytest.raises(RuntimeError):
            runner.run_queue(exp, [], [], [])
    assert calls == [("pe-plpr-aligng", 42)]
    assert json.loads((tmp_path / "queue-status.json").read_text())["status"] == (
        "interrupted" if outcome == "interrupt" else "failed"
    )


def test_partial_queue_resume_preserves_completed_and_current_state(tmp_path, monkeypatch):
    exp = study(tmp_path)
    calls = []

    def fake(model, seed, train, val, vectors, exp, directory, **kwargs):
        calls.append((model, seed))
        directory.mkdir(parents=True, exist_ok=True)
        if len(calls) == 2:
            (directory / "last.pt").write_bytes(b"resume-me")
            return None
        if (directory / "last.pt").exists():
            assert (directory / "last.pt").read_bytes() == b"resume-me"
        (directory / "best.pt").write_bytes(b"best")
        result = {
            "status": "PASS",
            "identity": {"experiment_sha256": digest(exp), "model": model, "seed": seed},
            "checkpoint_sha256": sha256(directory / "best.pt"),
        }
        atomic_write(directory / "result.json", result)
        return result

    monkeypatch.setattr(runner.engine, "train_run", fake)
    monkeypatch.setattr(runner, "summarize_matrix", lambda *_: None)
    assert runner.run_queue(exp, [], [], []) == 130
    assert runner.run_queue(exp, [], [], []) == 0
    assert calls == [runner.queue()[0], runner.queue()[1], *runner.queue()[1:]]


def test_signal_before_first_job_prevents_any_training(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No job may start after stop request")

    monkeypatch.setattr(runner.engine, "train_run", forbidden)
    assert runner.run_queue(study(tmp_path), [], [], [], stop_requested=lambda: True) == 130


@pytest.mark.parametrize("model", runner.MODELS)
@pytest.mark.parametrize("training", [True, False])
def test_probe_wiring_cpu_uses_actual_loss_and_gradients(tmp_path, model, training):
    exp = study(tmp_path)
    train, val, vectors = fixtures()
    result = probe.measure(model, train if training else val, training, exp, vectors, device="cpu", warmup=0, repeats=1)
    row = result["measurements"][0]
    assert result["status"] == "PASS" and row["loss"] > 0 and row["query_pairs"] > 0
    if training:
        assert row["gradient_norm"] > 0
    else:
        assert row["gradient_norm"] is None


def test_probe_gate_rejects_wrong_identity_or_missing_model(tmp_path):
    exp = study(tmp_path)
    rows = [
        {
            "model": m,
            "case": c,
            "images": 512,
            "status": "PASS",
            "variant": "fast",
            "original": {"status": "PASS"},
            "eval_equivalence": {"status": "PASS"},
        }
        for m in runner.MODELS
        for c in ("train_max_candidates", "train_max_sampled", "val_max_candidates")
    ]
    evidence = {"status": "PASS", "experiment_sha256": digest(exp), "results": rows}
    path = tmp_path / "capacity.json"
    atomic_write(path, evidence)
    runner.verify_probe(exp, path)
    evidence["results"] = rows[:-1]
    atomic_write(path, evidence)
    with pytest.raises(ValueError, match="Incomplete"):
        runner.verify_probe(exp, path)
    evidence["experiment_sha256"] = "changed"
    atomic_write(path, evidence)
    with pytest.raises(ValueError, match="probe"):
        runner.verify_probe(exp, path)


def test_shell_launcher_syntax_and_tmux_detachment_contract():
    import subprocess

    path = Path("scripts/run_aligng_gqa_fast.sh")
    subprocess.run(["bash", "-n", str(path)], check=True)
    content = path.read_text()
    assert "tmux new-session -d" in content and "--stage check" in content
    assert "--stage train-all >> logs/aligng-gqa-fast.log 2>&1" in content
