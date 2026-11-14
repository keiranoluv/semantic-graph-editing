"""Synthetic CPU checks for structured edits, leakage, learning and orchestration."""
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from src.editing.operations import Edge, EditGraph, Node, apply_operations, diff_aligned_graphs
from tools.edit_model_v1.codec import decode_actions, encode_actions, normalize_label, score_predictions
from tools.edit_model_v1.data import Example
from tools.edit_model_v1.model import collate_inputs
from tools.edit_model_v1.text import TextStore, load_store, text_key
from tools.emnlp2020_graph_modification_v1.dataset import read, sha256, write
from tools.magicbrush_experiment_v1.cache import input_texts
from tools.magicbrush_experiment_v3 import pipeline, training
from tools.magicbrush_experiment_v3.decoding import decode_batch
from tools.magicbrush_experiment_v3.model import StructuredPredictor
from tools.magicbrush_experiment_v3.targets import encode_targets, structured_loss
from tools.magicbrush_experiment_v3.vocabulary import build_vocabulary, coverage, prototype_inputs, prototype_vectors
from tools.prepare_magicbrush import digest


@pytest.fixture(autouse=True)
def cpu_runtime():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture(all_kinds=False):
    if all_kinds:
        source = EditGraph(
            (Node("a", "cup", ("red",)), Node("b", "table"), Node("c", "book")),
            (Edge("x", "a", "on", "b"), Edge("y", "c", "near", "b")),
        )
        target = EditGraph(
            (Node("a", "cup", ("blue",)), Node("b", "table"), Node("d", "đèn", ("green",))),
            (Edge("x", "a", "under", "b"), Edge("z", "d", "near", "b")),
        )
    else:
        source = EditGraph((Node("a", "cup", ("red",)),))
        target = EditGraph((Node("a", "cup", ("blue",)),))
    source, target, operations = normalize_label(source, target, diff_aligned_graphs(source, target))
    example = Example("toy:train", "train", "train", "Synthetic blue cup", source, target, operations)
    examples = {"train": [example], "val": [replace(example, id="toy:val", split="val", group="val")]}
    vocab = build_vocabulary(examples["train"])
    texts = input_texts(examples["train"] + examples["val"] + prototype_inputs(vocab))
    vectors = torch.nn.functional.normalize(
        torch.randn(len(texts), 512, generator=torch.Generator().manual_seed(7)), dim=-1
    )
    store = TextStore([text_key(t) for t in texts], vectors)
    config = read(Path("configs/level2/magicbrush-manual-experiment-v3.json"))
    config.update(epochs=6, minimum_epochs=4, patience=2, validation_interval=1, batch_size=1, val_batch_size=1)
    config["model"].update(hidden=16, layers=1, dropout=0.0)
    config["expected_counts"]["test"] = 1
    identity = {"synthetic_only": True, "vocabulary": vocab, "vocabulary_sha256": digest(vocab), "config": config}
    return examples, store, config, identity


def target_logits(targets, vocab, config):
    """Oracle logits ONLY test encode/decode roundtrip, never performance evidence."""
    result = {}
    for field, values in targets.items():
        if field.endswith("_attributes"):
            result[field] = torch.where(values == 1, 10.0, -10.0)
            continue
        suffix = field.rsplit("_", 1)[-1]
        classes = {
            "action": 3,
            "present": 2,
            "label": len(vocab["labels"]) + 1,
            "count": config["model"]["max_attributes"] + 1,
            "predicate": len(vocab["predicates"]) + 1,
            "subject": config["model"]["max_nodes"] + config["model"]["add_slots"] + 1,
            "object": config["model"]["max_nodes"] + config["model"]["add_slots"] + 1,
        }[suffix]
        selected = values.clone()
        selected[selected == -100] = 1 if field.startswith("new_") and suffix not in ("present", "count") else 0
        logits = torch.full((*values.shape, classes), -10.0)
        logits.scatter_(-1, selected.unsqueeze(-1), 10.0)
        result[field] = logits
    return result


def test_all_six_operations_roundtrip_through_slots_and_projection():
    examples, _, config, identity = fixture(all_kinds=True)
    e = examples["train"][0]
    assert len({op.kind for op in e.operations}) == 6
    targets = encode_targets([e], identity["vocabulary"], config)
    logits = target_logits(targets, identity["vocabulary"], config)
    rows = decode_batch([e.model_input()], logits, identity["vocabulary"], 20)
    for key in ("tokens", "raw_tokens"):
        metrics, errors = score_predictions([e], [rows[0][key]])
        assert errors == [None] and metrics["exact_target_graph_accuracy"] == 1
    assert rows[0]["constraints"] == {}
    target_blinded = replace(e, target=object(), operations=object())
    assert decode_batch([target_blinded.model_input()], logits, identity["vocabulary"], 20) == rows


def test_train_only_vocabulary_oov_retained_and_source_copy_supported():
    examples, _, config, identity = fixture()
    vocab = identity["vocabulary"]
    with pytest.raises(ValueError, match="training split"):
        build_vocabulary(examples["val"])
    source = EditGraph((Node("n001", "UNSEEN_SOURCE_LABEL", ("SOURCE_ONLY_ATTRIBUTE", "red")),))
    target = EditGraph((Node("n001", "UNSEEN_SOURCE_LABEL", ("SOURCE_ONLY_ATTRIBUTE", "blue")),))
    e = replace(examples["val"][0], source=source, target=target, operations=diff_aligned_graphs(source, target))
    targets = encode_targets([e], vocab, config)
    assert targets["node_label"][0, 0] == 0
    rows = decode_batch([e.model_input()], target_logits(targets, vocab, config), vocab, 20)
    assert score_predictions([e], [rows[0]["tokens"]])[0]["exact_target_graph_accuracy"] == 1
    target = EditGraph((Node("n001", "UNSEEN_TARGET_LABEL", ("blue",)),))
    e = replace(e, target=target, operations=diff_aligned_graphs(source, target))
    report = coverage([e], vocab)
    assert report["oov_payload_fields"] == 1 and report["excluded_examples"] == 0
    assert "UNSEEN_TARGET_LABEL" not in str(vocab)
    assert encode_targets([e], vocab, config)["node_label"][0, 0] == -100
    assert score_predictions([e], [encode_actions(())[1:]])[0]["examples"] == 1


def test_raw_invalid_decisions_and_explicit_constraint_effects_are_preserved():
    examples, _, config, identity = fixture(all_kinds=True)
    e = examples["train"][0]
    noop = replace(e, target=e.source, operations=())
    logits = target_logits(encode_targets([noop], identity["vocabulary"], config), identity["vocabulary"], config)
    logits["node_action"][0, 0] = torch.tensor([-10.0, 10.0, -10.0])  # delete a node but leave its incident edge
    row = decode_batch([e.model_input()], logits, identity["vocabulary"], 20)[0]
    assert score_predictions([e], [row["raw_tokens"]])[0]["invalid_transactions"] == 1
    operations = decode_actions(row["tokens"], e.source)
    apply_operations(e.source, operations)
    assert {o.kind for o in operations} == {"delete_node", "delete_edge"}
    assert row["constraints"]["incident_edges_explicitly_deleted"] == 1


def test_duplicate_and_dead_endpoint_constraints_do_not_use_target():
    examples, _, config, identity = fixture(all_kinds=True)
    e, vocab = examples["train"][0], identity["vocabulary"]
    noop = replace(e, target=e.source, operations=())
    logits = target_logits(encode_targets([noop], vocab, config), vocab, config)

    def choose(field, slot, category):
        logits[field][0, slot] = -10.0
        logits[field][0, slot, category] = 10.0

    # Predict a duplicate of the first source edge.
    old = e.source.edges[0]
    positions = {n.id: i + 1 for i, n in enumerate(e.source.nodes)}
    choose("new_edge_present", 0, 1)
    choose("new_edge_subject", 0, positions[old.subject])
    choose("new_edge_object", 0, positions[old.object])
    choose("new_edge_predicate", 0, vocab["predicates"].index(old.predicate) + 1)
    row = decode_batch([e.model_input()], logits, vocab, 20)[0]
    assert row["constraints"]["duplicate_edges_removed"] == 1
    assert decode_actions(row["tokens"], e.source) == ()
    assert score_predictions([e], [row["raw_tokens"]])[0]["invalid_transactions"] == 1

    # The new edge now selects a new-node slot whose presence head is off.
    choose("new_edge_subject", 0, 21)
    row = decode_batch([e.model_input()], logits, vocab, 20)[0]
    assert row["constraints"]["masked_dead_endpoint_selections"] == 1
    apply_operations(e.source, decode_actions(row["tokens"], e.source))


def test_attribute_edits_compose_an_unseen_combination():
    examples, _, config, identity = fixture()
    e, vocab = examples["val"][0], identity["vocabulary"]
    # Both atoms occur in train, but the joint tuple (blue, red) does not.
    target = EditGraph((Node("n001", "cup", ("red", "blue")),))
    e = replace(e, target=target, operations=diff_aligned_graphs(e.source, target))
    logits = target_logits(encode_targets([e], vocab, config), vocab, config)
    row = decode_batch([e.model_input()], logits, vocab, 20)[0]
    assert score_predictions([e], [row["tokens"]])[0]["exact_target_graph_accuracy"] == 1


@pytest.mark.parametrize("kind", ["mlp", "gnn"])
def test_real_model_backward_all_heads_and_synthetic_learning(tmp_path, kind):
    examples, store, config, identity = fixture(all_kinds=True)
    vocab = identity["vocabulary"]
    model = StructuredPredictor(
        kind=kind, vocabulary=vocab, prototypes=prototype_vectors(store, vocab), **config["model"]
    )
    logits = model(collate_inputs([examples["train"][0].model_input()], store))
    loss = structured_loss(logits, encode_targets(examples["train"], vocab, config), config)
    loss.backward()
    assert torch.isfinite(loss) and model.node_action.weight.grad.abs().sum() > 0
    assert model.subject_query.weight.grad.abs().sum() > 0
    # Actual learning and autonomous decoding of a repeated toy, no mock/oracle.
    examples, store, config, identity = fixture()
    config.update(epochs=100, minimum_epochs=20, patience=20, validation_interval=10, lr=0.01, weight_decay=0.0)
    result = training.train_one(examples, store, config, identity, kind, 42, tmp_path, device="cpu")
    assert result["validation"]["exact_target_graph_accuracy"] == 1
    assert result["quality_gate"]["passed"]


def test_interrupted_training_resumes_identical_cpu_state(tmp_path, monkeypatch):
    examples, store, config, identity = fixture()
    # Dropout exercises RNG restoration, not just deterministic updates.
    config["model"]["dropout"] = 0.1
    expected = training.train_one(examples, store, config, identity, "gnn", 42, tmp_path / "full", device="cpu")
    real_save = training.save_checkpoint

    def interrupted(path, payload):
        real_save(path, payload)
        if Path(path).name == "last.pt" and payload["epoch"] == 2:
            raise KeyboardInterrupt("synthetic checkpoint interruption")

    monkeypatch.setattr(training, "save_checkpoint", interrupted)
    with pytest.raises(KeyboardInterrupt):
        training.train_one(examples, store, config, identity, "gnn", 42, tmp_path / "resume", device="cpu")
    monkeypatch.setattr(training, "save_checkpoint", real_save)
    actual = training.train_one(examples, store, config, identity, "gnn", 42, tmp_path / "resume", device="cpu")
    assert actual["history"] == expected["history"]
    a = torch.load(tmp_path / "full/last.pt", weights_only=True)
    b = torch.load(tmp_path / "resume/last.pt", weights_only=True)
    assert all(torch.equal(v, b["model"][k]) for k, v in a["model"].items())
    assert torch.equal(a["rng"]["cpu"], b["rng"]["cpu"])


def fixture_predictions(model, examples, store, config, device, *, failed=False):
    # Deliberately controlled predictions test ONLY queue orchestration.
    tokens = [encode_actions(() if failed else e.operations)[1:] for e in examples]
    metrics, errors = score_predictions(examples, tokens)
    return metrics, [
        {
            "id": e.id,
            "tokens": t,
            "error": err,
            "raw_tokens": t,
            "raw_error": err,
            "constraints": {},
            "operation_count": 0 if failed else len(e.operations),
        }
        for e, t, err in zip(examples, tokens, errors, strict=True)
    ]


@pytest.mark.parametrize("fail_at", [0, 1, 3, None])
def test_both_pilots_run_then_gate_all_remaining_seeds_and_test(tmp_path, monkeypatch, fail_at):
    examples, store, config, identity = fixture()
    config.update(epochs=1, minimum_epochs=1)
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    calls, reads, interrupt_once = [], [], [fail_at is None]
    real_train, real_eval = training.train_one, pipeline.evaluate_one

    def cache(es, cfg, ident, directory, root, *, scope):
        if (directory / "manifest.json").exists():
            return load_store(directory, ident)
        texts = input_texts(list(es) + prototype_inputs(ident["vocabulary"]))
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
            raise KeyboardInterrupt("synthetic interruption after pilots")
        index = len(calls)
        calls.append((kind, seed))
        monkeypatch.setattr(training, "validate", lambda *a: fixture_predictions(*a, failed=index == fail_at))
        return real_train(es, st, cfg, ident, kind, seed, directory, device="cpu", wandb_mode="disabled")

    def heldout(paths, count, budget):
        output = tmp_path / config["output"]
        assert len(list(output.glob("*/*/completed-receipt.json"))) == 6
        assert (output / "selection.json").exists()
        reads.append(True)
        return [replace(examples["val"][0], id="toy:test", split="test", group="test")], {
            "evaluated_pairs": 1,
            "original_test_pairs": 1,
            "deferred_pairs": 0,
        }

    monkeypatch.setattr(pipeline, "build_cache", cache)
    monkeypatch.setattr(pipeline, "train_one", train)
    monkeypatch.setattr(pipeline, "load_heldout", heldout)
    monkeypatch.setattr(pipeline, "validate", fixture_predictions)
    monkeypatch.setattr(pipeline, "evaluate_one", lambda *a, **kw: real_eval(*a, **kw, device="cpu"))
    output = tmp_path / config["output"]
    if fail_at is not None:
        for _ in range(2):
            with pytest.raises(training.QualityGateError):
                pipeline.execute(config, examples, {}, identity, {}, wandb_mode="disabled")
            assert len(calls) == max(2, fail_at + 1) and not reads
            assert read(output / "progress.json")["stage"] == "quality_gate_failed"
            assert not (output / "selection.json").exists()
    else:
        with pytest.raises(KeyboardInterrupt):
            pipeline.execute(config, examples, {}, identity, {}, wandb_mode="disabled")
        assert calls == [("mlp", 42), ("gnn", 42)] and not reads
        pipeline.execute(config, examples, {}, identity, {}, wandb_mode="disabled")
        assert calls == [("mlp", 42), ("gnn", 42), ("mlp", 123), ("mlp", 3407), ("gnn", 123), ("gnn", 3407)]
        assert read(output / "progress.json")["stage"] == "complete"
        assert (output / "raw-and-constraint-diagnostics.json").exists()
        assert (output / "test-vocabulary-coverage.json").exists()
        checksum = sha256(output / "summary.json")
        pipeline.execute(config, examples, {}, identity, {}, wandb_mode="disabled")
        assert len(calls) == 6 and sha256(output / "summary.json") == checksum
        (output / "mlp/42/predictions-val.json").write_text("[]")
        with pytest.raises(ValueError, match="modified"):
            pipeline.execute(config, examples, {}, identity, {}, wandb_mode="disabled")
