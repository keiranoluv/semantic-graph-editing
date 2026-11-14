"""CPU regression tests: selection, resume, gating and the heldout barrier."""
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from src.editing.operations import EditGraph, Node, Operation, apply_operations
from tools.edit_model_v1.codec import encode_actions, score_predictions
from tools.edit_model_v1.data import Example
from tools.edit_model_v1.text import TextStore, load_store, text_key
from tools.emnlp2020_graph_modification_v1.dataset import read, sha256, write
from tools.magicbrush_experiment_v1.cache import input_texts
from tools.magicbrush_experiment_v2 import pipeline, training


@pytest.fixture(autouse=True)
def cpu_runtime():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture():
    source = EditGraph((Node("n001", "cup", ("red",)),))
    operations = (Operation("replace_node", source.nodes[0], Node("n001", "cup", ("blue",))),)
    e = Example(
        "synthetic:train",
        "train",
        "train",
        "Synthetic: blue cup",
        source,
        apply_operations(source, operations),
        operations,
    )
    examples = {"train": [e], "val": [replace(e, id="synthetic:val", split="val", group="val")]}
    texts = input_texts(examples["train"] + examples["val"])
    vectors = torch.randn(len(texts), 512, generator=torch.Generator().manual_seed(7))
    store = TextStore([text_key(t) for t in texts], vectors)
    config = read(Path("configs/level2/magicbrush-manual-experiment-v2.json"))
    config.update(
        epochs=6,
        minimum_epochs=4,
        patience=2,
        validation_interval=1,
        batch_size=1,
        val_batch_size=1,
        max_action_tokens=4,
    )
    config["model"].update(hidden=8, byte_dim=4, layers=1)
    config["expected_counts"]["test"] = 1
    return examples, store, config


def perfect_predictions(model, examples, store, config, device):
    # Deliberate oracle ONLY for testing queue control flow; not learning evidence.
    tokens = [encode_actions(e.operations)[1:] for e in examples]
    metrics, errors = score_predictions(examples, tokens)
    return metrics, [
        {"id": e.id, "tokens": t, "error": err} for e, t, err in zip(examples, tokens, errors, strict=True)
    ]


def test_selection_breaks_zero_f1_ties_without_overriding_f1():
    zero = {"micro": {"f1": 0.0}, "valid_transaction_rate": 0.0}
    assert training.selection_rank(zero, 2) > training.selection_rank(zero, 5)
    valid = {**zero, "valid_transaction_rate": 0.8}
    assert training.selection_rank(valid, 5) > training.selection_rank(zero, 1)
    positive = {"micro": {"f1": 0.1}, "valid_transaction_rate": 0.5}
    assert training.selection_rank(positive, 5) > training.selection_rank(valid, 1)
    with pytest.raises(ValueError, match="Nonfinite"):
        training.selection_rank(zero, float("nan"))


@pytest.mark.parametrize(
    "f1,valid,expected", [(0, 1, False), (1, 0, False), (0.01, 0.5, True), (0.009, 1, False), (1, float("nan"), False)]
)
def test_quality_gate_rejects_noop_and_decode_collapse(f1, valid, expected):
    _, _, config = fixture()
    metrics = {"micro": {"f1": f1}, "valid_transaction_rate": valid}
    assert training.quality_gate(metrics, config)["passed"] is expected
    if not expected:
        with pytest.raises(training.QualityGateError):
            training.require_quality(metrics, config)


def test_minimum_epochs_prevents_v1_premature_stop():
    config = {"minimum_epochs": 80, "patience": 40}
    assert not training.should_stop(11, 1, config)
    assert not training.should_stop(79, 1, config)
    assert training.should_stop(80, 1, config)
    assert not training.should_stop(80, 75, config)


def test_zero_f1_selects_later_checkpoint_when_validation_loss_improves(tmp_path, monkeypatch):
    examples, store, config = fixture()
    losses = iter([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    monkeypatch.setattr(training, "validation_token_loss", lambda *a: next(losses))
    result = training.train_one(examples, store, config, {"synthetic": True}, "mlp", 42, tmp_path, device="cpu")
    assert result["epochs_completed"] == result["selected_epoch"] == 6
    assert result["validation"]["micro"]["f1"] == 0
    assert result["status"] == "QUALITY_GATE_FAILED"


def test_interrupted_training_resumes_exact_cpu_weights(tmp_path, monkeypatch):
    examples, store, config = fixture()
    identity = {"synthetic": "resume regression", "config": config}
    expected = training.train_one(examples, store, config, identity, "gnn", 42, tmp_path / "full", device="cpu")
    real_save = training.save_checkpoint

    def interrupt(path, payload):
        real_save(path, payload)
        if Path(path).name == "last.pt" and payload["epoch"] == 2:
            raise KeyboardInterrupt("synthetic interruption after an atomic checkpoint")

    monkeypatch.setattr(training, "save_checkpoint", interrupt)
    with pytest.raises(KeyboardInterrupt):
        training.train_one(examples, store, config, identity, "gnn", 42, tmp_path / "resume", device="cpu")
    monkeypatch.setattr(training, "save_checkpoint", real_save)
    actual = training.train_one(examples, store, config, identity, "gnn", 42, tmp_path / "resume", device="cpu")
    assert actual["history"] == expected["history"]
    assert actual["selected_epoch"] == expected["selected_epoch"]
    a = torch.load(tmp_path / "full/last.pt", weights_only=True)
    b = torch.load(tmp_path / "resume/last.pt", weights_only=True)
    assert all(torch.equal(v, b["model"][k]) for k, v in a["model"].items())
    assert torch.equal(a["rng"]["cpu"], b["rng"]["cpu"])


@pytest.mark.parametrize("fail_at", [0, 1, 3, None])
def test_queue_gates_pilots_and_every_seed_before_test(tmp_path, monkeypatch, fail_at):
    examples, store, config = fixture()
    config.update(epochs=1, minimum_epochs=1)
    identity = {"synthetic_control_flow_only": True, "config": config}
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    real_train = training.train_one
    real_validate = training.validate
    real_eval = pipeline.evaluate_one
    calls, test_reads = [], []
    interrupt_once = [fail_at is None]

    def cache(es, cfg, ident, directory, root, *, scope):
        if (directory / "manifest.json").exists():
            return load_store(directory, ident)
        texts = input_texts(es)
        vectors = torch.stack([store[t] for t in texts])
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(vectors, directory / "vectors.pt")
        meta = {
            "identity": ident,
            "vectors_sha256": sha256(directory / "vectors.pt"),
            "shape": list(vectors.shape),
            "text_sha256": [text_key(t) for t in texts],
            "scope": scope,
        }
        write(directory / "manifest.json", meta)
        return load_store(directory, ident)

    def train(es, st, cfg, ident, kind, seed, directory, **kwargs):
        if len(calls) == 2 and interrupt_once[0]:
            interrupt_once[0] = False
            raise KeyboardInterrupt("synthetic queue interruption after both pilots")
        index = len(calls)
        calls.append((kind, seed))
        monkeypatch.setattr(training, "validate", real_validate if index == fail_at else perfect_predictions)
        return real_train(es, st, cfg, ident, kind, seed, directory, device="cpu", wandb_mode="disabled")

    def heldout(paths, count, budget):
        output = tmp_path / config["output"]
        assert len(list(output.glob("*/*/completed-receipt.json"))) == 6
        assert (output / "selection.json").exists()
        test_reads.append(True)
        return [replace(examples["val"][0], id="synthetic:test", split="test", group="test")], {
            "evaluated_pairs": 1,
            "original_test_pairs": 1,
            "deferred_pairs": 0,
        }

    monkeypatch.setattr(pipeline, "build_cache", cache)
    monkeypatch.setattr(pipeline, "train_one", train)
    monkeypatch.setattr(pipeline, "load_heldout", heldout)
    monkeypatch.setattr(pipeline, "validate", perfect_predictions)
    monkeypatch.setattr(pipeline, "evaluate_one", lambda *a, **kw: real_eval(*a, **kw, device="cpu"))
    output = tmp_path / config["output"]
    if fail_at is not None:
        with pytest.raises(training.QualityGateError):
            pipeline.execute(config, examples, {}, identity, {}, wandb_mode="disabled")
        assert len(calls) == fail_at + 1 and not test_reads
        assert read(output / "progress.json")["stage"] == "quality_gate_failed"
        assert not (output / "selection.json").exists()
        with pytest.raises(training.QualityGateError):
            pipeline.execute(config, examples, {}, identity, {}, wandb_mode="disabled")
        assert len(calls) == fail_at + 1 and not test_reads
    else:
        with pytest.raises(KeyboardInterrupt):
            pipeline.execute(config, examples, {}, identity, {}, wandb_mode="disabled")
        assert calls == [("mlp", 42), ("gnn", 42)] and not test_reads
        pipeline.execute(config, examples, {}, identity, {}, wandb_mode="disabled")
        assert calls == [("mlp", 42), ("gnn", 42), ("mlp", 123), ("mlp", 3407), ("gnn", 123), ("gnn", 3407)]
        assert read(output / "progress.json")["stage"] == "complete"
        assert "token loss" in read(output / "selection.json")["selection"]
        checksum = sha256(output / "summary.json")
        pipeline.execute(config, examples, {}, identity, {}, wandb_mode="disabled")
        assert len(calls) == 6 and sha256(output / "summary.json") == checksum
        (output / "mlp/42/predictions-val.json").write_text("[]")
        with pytest.raises(ValueError, match="modified"):
            pipeline.execute(config, examples, {}, identity, {}, wandb_mode="disabled")


def test_config_disallows_old_output_and_zero_quality_thresholds():
    config = read(Path("configs/level2/magicbrush-manual-experiment-v2.json"))
    pipeline.validate_config(config, Path.cwd())
    config["output"] = "runs/level2/magicbrush-manual-experiment-v1"
    with pytest.raises(ValueError, match="isolated"):
        pipeline.validate_config(config, Path.cwd())
    config["output"] = "runs/level2/magicbrush-manual-experiment-v2"
    config["quality_gate"]["minimum_micro_f1"] = 0
    with pytest.raises(ValueError, match="thresholds"):
        pipeline.validate_config(config, Path.cwd())


@pytest.mark.parametrize("kind", ["mlp", "gnn"])
def test_real_decoder_can_overfit_one_synthetic_transaction_on_cpu(tmp_path, kind):
    # A deliberately repeated toy example checks learning/decoding, NOT heldout
    # generalization. No mocks or reference-guided generation in this test.
    examples, store, config = fixture()
    config.update(
        epochs=200,
        minimum_epochs=40,
        patience=40,
        validation_interval=20,
        max_action_tokens=128,
        lr=0.01,
        weight_decay=0.0,
    )
    config["model"].update(hidden=32, byte_dim=16)
    result = training.train_one(
        examples, store, config, {"synthetic_overfit_only": True}, kind, 42, tmp_path, device="cpu"
    )
    assert result["validation"]["exact_operation_set_accuracy"] == 1
    assert result["validation"]["valid_transaction_rate"] == 1
    assert result["quality_gate"]["passed"]
