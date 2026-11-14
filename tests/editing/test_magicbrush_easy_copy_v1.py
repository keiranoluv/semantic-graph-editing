"""CPU regression, open-vocabulary, gradient, learning and queue integrity tests."""
from dataclasses import replace

import pytest
import torch
from test_magicbrush_experiment_v3 import fixture, target_logits

from src.editing.operations import Edge, EditGraph, Node, diff_aligned_graphs
from tools.edit_model_v1.codec import decode_actions, normalize_label, score_predictions
from tools.edit_model_v1.data import Example
from tools.edit_model_v1.text import TextStore, text_key
from tools.emnlp2020_graph_modification_v1.dataset import read
from tools.magicbrush_easy_copy_v1 import runner
from tools.magicbrush_easy_copy_v1.candidates import candidates
from tools.magicbrush_easy_copy_v1.features import GroundingStore, cache_inputs, collate_inputs
from tools.magicbrush_easy_copy_v1.model import StructuredPredictor
from tools.magicbrush_easy_copy_v1.projection import project_example
from tools.magicbrush_easy_copy_v1.runtime import decode_batch, targets_for, training_step
from tools.magicbrush_easy_copy_v1.training import train_one
from tools.magicbrush_experiment_v3.cache import input_texts
from tools.magicbrush_experiment_v3.targets import structured_loss
from tools.magicbrush_experiment_v3.vocabulary import build_vocabulary, prototype_inputs, prototype_vectors
from tools.magicbrush_lexical_pilot_v1.model import StructuredPredictor as PreviousPredictor


@pytest.fixture(autouse=True)
def cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def example(source, target, instruction):
    source, target, ops = normalize_label(source, target, diff_aligned_graphs(source, target))
    return Example("toy", "train", "group", instruction, source, target, ops)


def wrapped(es, vocab):
    items = es["train"] + es["val"]
    texts = input_texts(items + cache_inputs(items) + prototype_inputs(vocab))
    vectors = torch.nn.functional.normalize(
        torch.randn(len(texts), 512, generator=torch.Generator().manual_seed(7)), dim=-1
    )
    return GroundingStore(TextStore([text_key(t) for t in texts], vectors), vocab)


def model(store, cfg, *, copy=True, kind="mlp"):
    return StructuredPredictor(
        kind=kind,
        use_copy=copy,
        vocabulary=store.vocabulary,
        prototypes=prototype_vectors(store, store.vocabulary),
        **cfg["model"],
    )


@pytest.mark.parametrize(
    "before,after,prompt",
    [
        (("application windows",), ("car image",), "put a car on the screen of the laptop"),
        (("facing right", "head resting"), ("facing camera", "head raised"), "the dog is looking forward"),
        (("sparse",), ("dense",), "let trees border the zoo"),
        (("gray desktop",), ("red header", "webpage"), "let the laptop have a red web page"),
    ],
)
def test_property_semantics_never_erased(before, after, prompt):
    e = example(EditGraph((Node("a", "object", before),)), EditGraph((Node("a", "object", after),)), prompt)
    projected, audit = project_example(e)
    assert projected == e
    assert projected.operations and not audit["omitted_new_node_appearance"]


def test_optional_new_appearance_only_and_idempotence():
    source = EditGraph((Node("a", "table", ("wooden",)), Node("b", "book")), (Edge("x", "b", "on", "a"),))
    target = EditGraph(
        (Node("a", "table", ("wooden",)), Node("c", "laptop", ("black", "open"))), (Edge("y", "c", "on", "a"),)
    )
    e = example(source, target, "Put a laptop on the table")
    p, audit = project_example(e)
    assert p.source == e.source and len(p.operations) == len(e.operations)
    assert next(n for n in p.target.nodes if n.label == "laptop").attributes == ()
    assert project_example(p)[0] == p and audit["omitted_new_node_appearance"]
    assert project_example(replace(e, instruction="Put an open black laptop on the table"))[0].target == e.target


def test_duplicate_new_instances_retain_identity_properties():
    e = example(EditGraph(), EditGraph((Node("a", "duck", ("yellow",)), Node("b", "duck", ("white",)))), "Add ducks")
    assert project_example(e)[0] == e


def test_candidates_are_literal_unseen_spans_not_closed_vocabulary():
    assert "gorilla" in candidates("Have there be a gorilla crossing the road")
    assert "guitar" in candidates("Have the gorilla be playing a guitar")
    assert "pizza cutter" in candidates("Add a pizza cutter")
    assert "can" not in candidates("Can you add a quokka?")
    assert "quokka" in candidates("Can you add a quokka?")
    assert "car" not in candidates("add a carpet")
    assert len(candidates(" ".join("noun" + str(i) for i in range(100)))) <= 64


@pytest.mark.parametrize("kind", ["mlp", "gnn"])
def test_no_copy_parity_and_input_blinding(kind):
    es, _, cfg, identity = fixture(all_kinds=True)
    store = wrapped(es, identity["vocabulary"])
    torch.manual_seed(42)
    old = PreviousPredictor(
        kind=kind, vocabulary=store.vocabulary, prototypes=prototype_vectors(store, store.vocabulary), **cfg["model"]
    ).eval()
    torch.manual_seed(42)
    new = model(store, cfg, copy=False, kind=kind).eval()
    batch = collate_inputs([es["train"][0].model_input()], store)
    a, b = old(batch), new(batch)
    assert all(torch.equal(v, b[k]) for k, v in a.items())
    blinded = replace(es["train"][0], target=object(), operations=object())
    changed = collate_inputs([blinded.model_input()], store)
    assert all(torch.equal(v, changed[k]) for k, v in batch.items())
    with pytest.raises(ValueError, match="Only source"):
        collate_inputs([{**blinded.model_input(), "target": object()}], store)


def test_copy_loss_trains_query_and_bias_on_in_vocabulary_mentions():
    es, _, cfg, identity = fixture()
    e = example(
        EditGraph((Node("a", "table"),)), EditGraph((Node("a", "table"), Node("b", "cup"))), "Add a cup on the table"
    )
    es = {"train": [e], "val": [replace(e, split="val")]}
    vocab = build_vocabulary(es["train"])
    store = wrapped(es, vocab)
    m = model(store, cfg)
    inputs = collate_inputs([e.model_input()], store)
    targets = targets_for([e], vocab, cfg, use_copy=True)
    result = m(inputs)
    structured_loss(result, targets, cfg).backward()
    assert m.copy_bias.grad[1].abs() > 0 and torch.isfinite(m.copy_bias.grad).all()
    assert m.label.query.weight.grad.abs().sum() > 0
    # A duplicate spelling contributes through marginalization, not two labels.
    offset = 1 + len(vocab["labels"])
    idx = candidates(e.instruction).index("cup")
    assert (result["new_node_label"][..., offset + idx] == -1e4).all()


@pytest.mark.parametrize("operation", ["add", "replace"])
def test_unseen_copy_target_roundtrip_is_not_added_to_train_vocab(operation):
    es, _, cfg, identity = fixture()
    vocab = identity["vocabulary"]
    source = EditGraph((Node("n001", "cup", ("red",)),))
    target = (
        EditGraph(source.nodes + (Node("newn001", "quokka"),))
        if operation == "add"
        else EditGraph((Node("n001", "quokka"),))
    )
    e = example(source, target, "Add a quokka" if operation == "add" else "Replace the cup with a quokka")
    targets = targets_for([e], vocab, cfg, use_copy=True)
    local = {**vocab, "labels": vocab["labels"] + list(candidates(e.instruction))}
    logits = target_logits(targets, local, cfg)
    row = decode_batch([e.model_input()], logits, vocab, 20, use_copy=True)[0]
    measured, errors = score_predictions([e], [row["tokens"]])
    assert errors == [None] and measured["exact_target_graph_accuracy"] == 1
    assert "quokka" not in vocab["labels"]


@pytest.mark.parametrize("kind", ["mlp", "gnn"])
def test_learning_and_completed_resume(tmp_path, kind):
    es, _, cfg, identity = fixture()
    store = wrapped(es, identity["vocabulary"])
    cfg.update(epochs=50, minimum_epochs=10, patience=8, lr=0.01, use_copy=True)
    result = train_one(es, store, cfg, identity, kind, 42, tmp_path, device="cpu")
    assert result["validation"]["exact_target_graph_accuracy"] == 1
    assert train_one(es, store, cfg, identity, kind, 42, tmp_path, device="cpu") == result


def test_variable_candidate_padding_empty_and_real_training_gradients():
    es, _, cfg, identity = fixture(all_kinds=True)
    e = es["train"][0]
    es["train"] += [
        replace(e, id="empty", instruction=""),
        replace(e, id="long", instruction="Add a golden quokka on a table"),
    ]
    store = wrapped(es, identity["vocabulary"])
    m = model(store, cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=0.001)
    loss, count = training_step(m, opt, es["train"], store, cfg, "cpu")
    assert count == 3 and loss > 0
    outputs = m(collate_inputs([x.model_input() for x in es["train"]], store))
    rows = decode_batch([x.model_input() for x in es["train"]], outputs, store.vocabulary, 20, use_copy=True)
    assert len(rows) == 3


def test_queue_runs_four_matched_arms_resumes_and_detects_tampering(tmp_path, monkeypatch):
    es, _, cfg, identity = fixture()
    cfg.update(
        seed=42,
        output="runs/test",
        epochs=1,
        minimum_epochs=1,
        patience=1,
        queue=[{"variant": v, "model": m} for v in ("no_copy", "copy") for m in ("mlp", "gnn")],
    )
    store = wrapped(es, identity["vocabulary"])
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    first = runner.run_queue(cfg, es, store, identity, device="cpu")
    assert len(first["models"]) == 4 and first["test_used"] is False
    assert runner.run_queue(cfg, es, store, identity, device="cpu") == first
    p = tmp_path / "runs/test/copy/mlp/42/predictions-val.json"
    p.write_text("[]")
    with pytest.raises(ValueError, match="modified"):
        runner.run_queue(cfg, es, store, identity, device="cpu")


@pytest.mark.parametrize(
    "prompt,attributes",
    [
        ("Add a sleeping person to the couch", ("eyes closed", "lying on side")),
        ("Have there be a painting of a flower on the wall", ("white flower", "yellow center")),
    ],
)
def test_requested_new_state_and_depiction_content_are_protected(prompt, attributes):
    e = example(EditGraph(), EditGraph((Node("a", "object", attributes),)), prompt)
    assert project_example(e)[0] == e


def test_model_learns_to_emit_name_outside_fixed_vocabulary(tmp_path):
    es, _, cfg, identity = fixture()
    # Deliberately withhold the only target name from the fixed vocabulary to
    # force the learned copy route. This is a synthetic mechanism check only.
    e = example(EditGraph((Node("a", "cup"),)), EditGraph((Node("a", "cup"), Node("b", "quokka"))), "Add a quokka")
    es = {"train": [e], "val": [replace(e, split="val", id="val", group="val")]}
    store = wrapped(es, identity["vocabulary"])
    cfg.update(epochs=80, minimum_epochs=15, patience=10, lr=0.01, use_copy=True)
    result = train_one(es, store, cfg, identity, "mlp", 42, tmp_path, device="cpu")
    assert result["validation"]["exact_target_graph_accuracy"] == 1
    assert "quokka" not in identity["vocabulary"]["labels"]
    ops = decode_actions(read(tmp_path / "predictions-val.json")[0]["tokens"], e.source)
    assert any(op.after and getattr(op.after, "label", None) == "quokka" for op in ops)
