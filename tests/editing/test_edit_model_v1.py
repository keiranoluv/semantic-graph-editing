"""Synthetic CPU-only learning fixtures; never evidence for real model quality."""

import json

import pytest
import torch

from src.editing.operations import Edge, EditGraph, Node, Operation, apply_operations, diff_aligned_graphs
from src.knowledge.transe.core import sha256
from tools.edit_model_v1 import codec, data, runner
from tools.edit_model_v1.model import EditPredictor, collate_inputs, collate_targets
from tools.edit_model_v1.text import TextStore, collect_texts, text_key
from tools.prepare_magicbrush import digest, encode
from tools.review_magicbrush import AXES, export_reviews


@pytest.fixture(autouse=True)
def cpu_runtime():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture_example(identifier="synthetic", split="train"):
    source = EditGraph((Node("a", "cup", ("red",)), Node("b", "table")), (Edge("on", "a", "on", "b"),))
    target = EditGraph((Node("a", "cup", ("blue",)), Node("b", "table")), source.edges)
    source, target, actions = codec.normalize_label(source, target, diff_aligned_graphs(source, target))
    return data.Example(
        identifier, split, identifier, "Synthetic fixture: change the cup to blue.", source, target, actions
    )


def fixture_store(examples):
    texts = collect_texts(examples)
    generator = torch.Generator().manual_seed(7)
    vectors = torch.nn.functional.normalize(torch.randn(len(texts), 16, generator=generator), dim=-1)
    return TextStore([text_key(t) for t in texts], vectors)


def fixture_dataset(tmp_path, count=20):
    rows, labels, cases = [], [], []
    for i in range(count):
        example = fixture_example(f"synthetic:{i}")
        split = "train" if i < count - 2 else "val" if i == count - 2 else "test"
        row = {
            "id": example.id,
            "img_id": str(i),
            "turn_index": 1,
            "source_group_id": f"group:{i}",
            "split": split,
            "official_split": "train",
            "instruction": example.instruction,
            "source_image": {"sha256": f"s{i}", "pixel_sha256": f"ps{i}"},
            "target_image": {"sha256": f"t{i}", "pixel_sha256": f"pt{i}"},
        }
        label = {
            "id": example.id,
            "split": split,
            "source_group_id": row["source_group_id"],
            "instruction": row["instruction"],
            "status": "candidate",
            "eligible_for_training": True,
            "problems": [],
            "source_graph": example.source.to_dict(),
            "target_graph": example.target.to_dict(),
            "operations": [a.to_dict() for a in example.operations],
        }
        evidence = digest({"source_sha256": f"s{i}", "target_sha256": f"t{i}", "label": label})
        rows.append(row)
        labels.append(label)
        if split == "train":
            cases.append({**row, "label": label, "graphs_ready": True, "evidence_sha256": evidence})
    manifest = {
        "records": rows,
        "splits": {k: [r["id"] for r in rows if r["split"] == k] for k in ("train", "val", "test")},
        "review_ids": [r["id"] for r in cases],
    }
    manifest_path, edits_path = tmp_path / "manifest.json", tmp_path / "edits.json"
    manifest_path.write_bytes(encode(manifest))
    edits_path.write_bytes(encode({"dataset_manifest_sha256": sha256(manifest_path), "records": labels}))
    packet = {"dataset_manifest_sha256": sha256(manifest_path), "cases": cases}
    packet["packet_sha256"] = digest(packet)
    packet_path = tmp_path / "packet.json"
    packet_path.write_bytes(encode(packet))
    reviews = {
        r["id"]: {
            "decision": "correct",
            "reviewer": "SYNTHETIC TEST REVIEWER",
            "notes": "fixture only",
            **{k: "correct" for k in AXES},
        }
        for r in cases
    }
    csv_path = tmp_path / "synthetic-review.csv"
    csv_path.write_bytes(export_reviews(packet, reviews))
    return manifest_path, edits_path, packet_path, csv_path


def test_all_six_operations_roundtrip_with_full_payload_and_new_unicode():
    source = EditGraph(
        (Node("a", "cup"), Node("b", "table"), Node("c", "book")),
        (Edge("x", "a", "on", "b"), Edge("y", "c", "near", "b")),
    )
    target = EditGraph(
        (Node("a", "cup", ("blue",)), Node("b", "table"), Node("d", "đèn 🟢")),
        (Edge("x", "a", "under", "b"), Edge("z", "d", "near", "b")),
    )
    source, target, operations = codec.normalize_label(source, target, diff_aligned_graphs(source, target))
    assert {op.kind for op in operations} == set(codec.KINDS)
    decoded = codec.decode_actions(codec.encode_actions(operations)[1:], source)
    assert decoded == operations and apply_operations(source, decoded) == target


def test_new_entity_ids_and_operation_order_are_independently_canonical():
    source = EditGraph((Node("n001", "table"),))
    first = (
        Operation("add_node", after=Node("arbitrary", "cup")),
        Operation("add_edge", after=Edge("edge-x", "arbitrary", "on", "n001")),
    )
    second = (
        Operation("add_edge", after=Edge("other-edge", "renamed", "on", "n001")),
        Operation("add_node", after=Node("renamed", "cup")),
    )
    assert codec.canonical_actions(source, first) == codec.canonical_actions(source, second)
    result = codec.canonical_actions(source, first)
    example = data.Example("synthetic", "val", "g", "add cup", source, apply_operations(source, result), result)
    metrics, errors = codec.score_predictions([example], [codec.encode_actions(second)[1:]])
    assert metrics["micro"]["f1"] == 1 and metrics["exact_target_graph_accuracy"] == 1 and errors == [None]


def test_unresolved_new_instance_identity_is_explicit():
    with pytest.raises(ValueError, match="unsupported_ambiguous_new_instance_identity"):
        codec.canonical_actions(
            EditGraph(), (Operation("add_node", after=Node("a", "cup")), Operation("add_node", after=Node("b", "cup")))
        )


def test_structural_validity_is_distinct_from_unsupported_predicted_identity():
    source = EditGraph()
    expected = (Operation("add_node", after=Node("newn001", "cup")),)
    example = data.Example("synthetic", "val", "g", "add cup", source, apply_operations(source, expected), expected)
    prediction = (Operation("add_node", after=Node("a", "cup")), Operation("add_node", after=Node("b", "cup")))
    metrics, errors = codec.score_predictions([example], [codec.encode_actions(prediction)[1:]])
    assert errors[0].startswith("identity:") and metrics["unsupported_predicted_identities"] == 1
    assert metrics["valid_transaction_rate"] == 1 and metrics["exact_target_graph_accuracy"] == 0


@pytest.mark.parametrize("tokens", [[], [codec.EOS], [100], [codec.PAD, codec.EOS], [258, codec.EOS]])
def test_invalid_decoding_is_failure_not_valid_noop(tokens):
    metrics, errors = codec.score_predictions([fixture_example()], [tokens])
    assert errors[0] and metrics["failed_predictions"] == 1
    assert metrics["valid_transaction_rate"] == metrics["micro"]["f1"] == 0


def test_duplicate_actions_are_false_positive_and_invalid_transaction():
    example = fixture_example()
    tokens = codec.encode_actions(example.operations * 2)[1:]
    metrics, errors = codec.score_predictions([example], [tokens])
    assert metrics["micro"]["tp"] == metrics["micro"]["fp"] == 1
    assert metrics["valid_transaction_rate"] == 0 and errors[0]


def test_loader_never_decodes_test_targets_or_encodes_target_text(tmp_path):
    manifest, edits, _, _ = fixture_dataset(tmp_path)
    payload = json.loads(edits.read_text())
    payload["records"][-1]["target_graph"] = "INVALID TEST LABEL MUST NOT BE DECODED"
    edits.write_bytes(encode(payload))
    examples, report = data.load_examples(manifest, edits)
    assert len(examples["train"]) == 18 and len(examples["val"]) == 1 and report["test_labels_decoded"] is False
    example = examples["train"][0]
    assert set(example.model_input()) == {"source", "instruction"}
    strings = collect_texts(examples)
    assert '["cup", ["blue"]]' not in strings and '["cup", ["red"]]' in strings


def test_leakage_and_corrupt_label_rejected(tmp_path):
    manifest, edits, _, _ = fixture_dataset(tmp_path)
    m = json.loads(manifest.read_text())
    m["records"][-1]["img_id"] = m["records"][0]["img_id"]
    manifest.write_bytes(encode(m))
    p = json.loads(edits.read_text())
    p["dataset_manifest_sha256"] = sha256(manifest)
    edits.write_bytes(encode(p))
    with pytest.raises(ValueError, match="leakage"):
        data.load_examples(manifest, edits)


def test_long_validation_labels_stay_in_denominator(tmp_path):
    manifest, edits, _, _ = fixture_dataset(tmp_path)
    examples, report = data.load_examples(manifest, edits, max_action_tokens=2)
    assert len(examples["train"]) == 0 and len(examples["val"]) == 1
    assert report["validation_labels_exceeding_decode_budget"] == [examples["val"][0].id]


def test_human_gate_requires_real_decisions_and_matching_per_pair_evidence(tmp_path):
    paths = fixture_dataset(tmp_path, count=22)
    gate = data.review_gate(*paths)
    assert gate["ready"] and gate["determinate_reviews"] == 20 and not gate["automatically_approved"]
    packet = json.loads(paths[2].read_text())
    paths[3].write_bytes(export_reviews(packet, {}))
    assert not data.review_gate(*paths)["ready"]


def test_human_reviews_can_follow_unchanged_pair_evidence_to_full_export(tmp_path):
    paths = fixture_dataset(tmp_path, count=22)
    payload = json.loads(paths[1].read_text())
    payload["records"][-1]["status"] = "pending_graphs"  # Only nonreviewed test metadata changed.
    paths[1].write_bytes(encode(payload))
    assert data.review_gate(*paths)["ready"]
    payload["records"][0]["operations"][0]["after"]["attributes"] = ["green"]
    paths[1].write_bytes(encode(payload))
    with pytest.raises(ValueError, match="per-pair evidence"):
        data.review_gate(*paths)


def test_rejected_human_pair_is_excluded_and_counted(tmp_path):
    manifest, edits, _, _ = fixture_dataset(tmp_path)
    examples, report = data.load_examples(manifest, edits, rejected=["synthetic:0"])
    assert len(examples["train"]) == 17 and report["excluded"]["train:human_incorrect_or_uncertain"] == 1


@pytest.mark.parametrize("kind", ["mlp", "gnn"])
def test_model_has_finite_gradients_and_learns_on_cpu_without_text_gradients(kind):
    examples = {"train": [fixture_example()], "val": [fixture_example("val", "val")]}
    store = fixture_store(examples)
    inputs = collate_inputs([examples["train"][0].model_input()], store)
    previous, expected = collate_targets(examples["train"])
    torch.manual_seed(42)
    model = EditPredictor(kind=kind, text_dim=store.dim, hidden=32, byte_dim=16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    losses = [runner.training_step(model, optimizer, inputs, previous, expected, 5)[0] for _ in range(25)]
    assert losses[-1] < losses[0] * 0.4
    assert store.vectors.grad is None and not store.vectors.requires_grad
    alternate = previous.clone()
    alternate[:, 2:] = 50
    model.eval()
    with torch.inference_mode():
        assert torch.allclose(model(inputs, previous)[:, :2], model(inputs, alternate)[:, :2], atol=1e-6)
        first = model(inputs, previous[:, :1]).argmax(-1)[:, 0].tolist()
        assert [row[0] for row in model.predict(inputs, 1)] == first


def test_empty_source_and_isolated_nodes_have_finite_predictions():
    empty = data.Example(
        "empty",
        "train",
        "g",
        "add cup",
        EditGraph(),
        EditGraph((Node("newn001", "cup"),)),
        (Operation("add_node", after=Node("newn001", "cup")),),
    )
    examples = {"train": [empty, fixture_example()], "val": []}
    store = fixture_store(examples)
    batch = collate_inputs([e.model_input() for e in examples["train"]], store)
    model = EditPredictor(kind="gnn", text_dim=store.dim, hidden=32, byte_dim=16)
    assert torch.isfinite(model(batch, collate_targets(examples["train"])[0])).all()


def test_checkpoint_restores_exact_next_optimizer_step_on_cpu(tmp_path):
    example = fixture_example()
    examples = {"train": [example], "val": []}
    store = fixture_store(examples)
    inputs = collate_inputs([example.model_input()], store)
    previous, expected = collate_targets([example])
    torch.manual_seed(42)
    model = EditPredictor(kind="gnn", text_dim=store.dim, hidden=32, byte_dim=16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    runner.training_step(model, optimizer, inputs, previous, expected, 5)
    runner.save_checkpoint(
        tmp_path / "last.pt",
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": runner.capture_rng("cpu")},
    )
    expected_loss = runner.training_step(model, optimizer, inputs, previous, expected, 5)[0]
    resumed = EditPredictor(kind="gnn", text_dim=store.dim, hidden=32, byte_dim=16)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=0.001)
    checkpoint = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=True)
    resumed.load_state_dict(checkpoint["model"])
    resumed_optimizer.load_state_dict(checkpoint["optimizer"])
    runner.restore_rng(checkpoint["rng"], "cpu")
    assert expected_loss == runner.training_step(resumed, resumed_optimizer, inputs, previous, expected, 5)[0]
    assert all(torch.equal(p, q) for p, q in zip(model.parameters(), resumed.parameters(), strict=True))


def test_complete_cpu_fixture_training_resume_and_prediction_artifacts(tmp_path):
    examples = {"train": [fixture_example()], "val": [fixture_example("validation", "val")]}
    store = fixture_store(examples)
    config = json.loads(runner.DEFAULT_CONFIG.read_text())
    config.update(epochs=2, patience=2, max_action_tokens=4, batch_size=1, val_batch_size=1)
    config["model"].update(hidden=32, byte_dim=16)
    output = tmp_path / "synthetic-run"
    result = runner.train_one(examples, store, config, {"fixture": True}, "mlp", 42, output, device="cpu")
    assert result["status"] == "PASS" and result["test_used"] is False and result["text_encoder_frozen"] is True
    assert result["validation"]["failed_predictions"] == 1
    assert result["validation"]["valid_transaction_rate"] == 0
    assert result["no_edit_baseline_validation"]["valid_transaction_rate"] == 1
    again = runner.train_one(examples, store, config, {"fixture": True}, "mlp", 42, output, device="cpu")
    assert again == result and (output / "predictions-val.json").is_file()
    (output / "best.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checkpoint changed"):
        runner.train_one(examples, store, config, {"fixture": True}, "mlp", 42, output, device="cpu")


def test_main_preflight_blocks_before_any_cuda_call(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("GPU was touched during blocked preflight")

    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(
        runner, "preflight", lambda config: ({}, {"ready": False, "reason": "fixture missing labels"}, None)
    )
    assert runner.main(["--stage", "plan"]) == 2
    assert '"ready": false' in capsys.readouterr().out
    with pytest.raises(SystemExit, match="BLOCKED"):
        runner.main(["--stage", "train", "--model", "mlp", "--seed", "42"])


def test_interrupted_epoch_checkpoint_resumes_same_cpu_training(tmp_path, monkeypatch):
    examples = {"train": [fixture_example()], "val": [fixture_example("validation", "val")]}
    store = fixture_store(examples)
    config = json.loads(runner.DEFAULT_CONFIG.read_text())
    config.update(epochs=2, patience=2, max_action_tokens=4, batch_size=1, val_batch_size=1)
    config["model"].update(hidden=32, byte_dim=16)
    uninterrupted = runner.train_one(
        examples, store, config, {"fixture": True}, "gnn", 42, tmp_path / "continuous", device="cpu"
    )
    real_save = runner.save_checkpoint

    def interrupt_after_durable_epoch(path, payload):
        real_save(path, payload)
        if path.name == "last.pt":
            raise RuntimeError("simulated process interruption")

    monkeypatch.setattr(runner, "save_checkpoint", interrupt_after_durable_epoch)
    with pytest.raises(RuntimeError, match="simulated process"):
        runner.train_one(examples, store, config, {"fixture": True}, "gnn", 42, tmp_path / "resumed", device="cpu")
    monkeypatch.setattr(runner, "save_checkpoint", real_save)
    resumed = runner.train_one(
        examples, store, config, {"fixture": True}, "gnn", 42, tmp_path / "resumed", device="cpu"
    )
    assert resumed["history"] == uninterrupted["history"]
    first = torch.load(tmp_path / "continuous/last.pt", map_location="cpu", weights_only=True)
    second = torch.load(tmp_path / "resumed/last.pt", map_location="cpu", weights_only=True)
    assert all(torch.equal(first["model"][k], second["model"][k]) for k in first["model"])


def test_fixed_byte_vocabulary_needs_no_target_or_validation_vocabulary():
    assert codec.VOCAB_SIZE == 259
    config = json.loads(runner.DEFAULT_CONFIG.read_text())
    runner.validate_config(config)
    config["minimum_valid_pairs"] = 299
    with pytest.raises(ValueError, match=">=300"):
        runner.validate_config(config)


def test_source_fingerprinting_does_not_import_clip_or_query_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU source hashing must not query CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    fingerprints = runner.source_hashes()
    assert "installed_clip/clip.py" in fingerprints and "tools/edit_model_v1/model.py" in fingerprints
