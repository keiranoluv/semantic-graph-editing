"""Contract tests: real-data measurements live in experiment artifacts."""

from dataclasses import replace

import pytest
import torch

from src.editing.operations import Edge, EditGraph, Node, Operation, apply_operations
from tools.edit_model_v1.codec import decode_actions, encode_actions, score_predictions
from tools.edit_model_v1.data import Example
from tools.edit_model_v1.model import EditPredictor, collate_inputs
from tools.edit_model_v1.text import TextStore, text_key
from tools.emnlp2020_task_b_v2.data import filter_rows
from tools.emnlp2020_task_b_v2.model import ActionVocabulary, LexicalPredictor
from tools.emnlp2020_task_b_v2.runner import final_test


def example():
    source = EditGraph((Node("n001", "café"), Node("n002", "dog")), (Edge("e001", "n001", "near", "n002"),))
    operations = (Operation("replace_node", source.nodes[0], Node("n001", 'blue "car"')),)
    return Example("x", "train", "group", "make it blue", source, apply_operations(source, operations), operations)


def test_compact_codec_exact_roundtrip_unicode_and_before_from_source():
    e = example()
    vocabulary = ActionVocabulary.fit([e])
    actions = vocabulary.to_bytes(vocabulary.encode(e.operations)[1:])
    assert decode_actions(actions, e.source) == e.operations
    assert len(actions) == len(encode_actions(e.operations)) - 1
    assert score_predictions([e], [actions])[0]["exact_target_graph_accuracy"] == 1
    changed = replace(e, target=EditGraph((Node("other", "unseen"),), ()))
    assert changed.model_input() == e.model_input()


def test_codec_oov_is_counted_as_failure_not_fitted_on_validation():
    e = example()
    vocabulary = ActionVocabulary.fit([e])
    ops = (Operation("replace_node", e.source.nodes[0], Node("n001", "heldout-secret")),)
    assert 3 in vocabulary.encode(ops)
    encoded = vocabulary.to_bytes(vocabulary.encode(ops)[1:])
    assert (
        score_predictions([replace(e, operations=ops, target=apply_operations(e.source, ops))], [encoded])[0][
            "decode_failures"
        ]
        == 1
    )
    assert '"heldout-secret"' not in vocabulary.ids


def test_all_six_kinds_codec_without_fabricating_support():
    e = example()
    ops = (
        Operation("delete_edge", e.source.edges[0], None),
        Operation("delete_node", e.source.nodes[1], None),
        Operation("replace_node", e.source.nodes[0], Node("n001", "blue")),
        Operation("add_node", None, Node("newn001", "cat")),
        Operation("add_edge", None, Edge("newe001", "n001", "on", "newn001")),
    )
    vocabulary = ActionVocabulary.fit([replace(e, operations=ops)])
    assert decode_actions(vocabulary.to_bytes(vocabulary.encode(ops)[1:]), e.source) == ops
    relation = (Operation("replace_edge", e.source.edges[0], Edge("e001", "n001", "under", "n002")),)
    vocabulary = ActionVocabulary.fit([replace(e, operations=relation)])
    assert decode_actions(vocabulary.to_bytes(vocabulary.encode(relation)[1:]), e.source) == relation
    assert vocabulary.to_bytes([vocabulary.ids['"replace_edge"'], 2]) == []


def raw(label, identifier=0):
    return {
        "directed": False,
        "nodes": [{"id": identifier, "data": {"feature": label}}],
        "edges": [],
        "graph_metadata": {},
    }


def row(split, label, instruction="remove it", status="candidate"):
    return {"raw_source": raw(label), "split": split, "status": status, "instruction": instruction}


def test_original_split_group_filter_reserves_even_rejected_sources():
    rows = [
        row("train", "cat", status="rejected"),
        row("train", "dog"),
        row("train", "dog"),
        row("val", "cat"),
        row("val", "horse"),
    ]
    groups = filter_rows(rows)
    assert [r["exclusion"] for r in rows] == [
        "conversion_rejected",
        None,
        "duplicate_source_instruction_within_split",
        "source_overlap_earlier_official_split",
        None,
    ]
    test = [row("test", "horse"), row("test", "bird")]
    filter_rows(test, groups)
    assert test[0]["exclusion"] == "source_overlap_earlier_official_split"
    assert test[1]["exclusion"] is None


@pytest.mark.parametrize("kind", ["mlp", "gnn"])
def test_encoder_padding_optimization_matches_parent_and_cpu_gradient(kind):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    e = example()
    texts = [e.instruction, '["café", []]', '["dog", []]', "near"]
    store = TextStore([text_key(t) for t in texts], torch.randn(4, 8))
    batch = collate_inputs([e.model_input()], store, max_nodes=2, max_edges=1)
    vocabulary = ActionVocabulary.fit([e])
    model = LexicalPredictor(
        vocabulary_size=vocabulary.size, kind=kind, text_dim=8, hidden=16, byte_dim=48, max_nodes=2, max_edges=1
    )
    parent = EditPredictor(kind=kind, text_dim=8, hidden=16, byte_dim=48, max_nodes=2, max_edges=1)
    parent.load_state_dict(
        {
            k: v
            for k, v in model.state_dict().items()
            if k not in ("byte_embedding.weight", "output.weight", "output.bias")
        },
        strict=False,
    )
    for expected, actual in zip(parent.encode(batch), model.encode(batch), strict=True):
        assert torch.equal(expected, actual)
    previous, truth = vocabulary.collate([e])
    loss = torch.nn.functional.cross_entropy(model(batch, previous).flatten(0, 1), truth.flatten())
    loss.backward()
    assert torch.isfinite(loss)
    assert model.text_projection[0].weight.grad.abs().sum() > 0


def test_test_cannot_open_before_complete_freeze(tmp_path):
    with pytest.raises(FileNotFoundError, match="frozen-checkpoints"):
        final_test({"output": str(tmp_path)}, tmp_path / "never-open-archive.tgz")


def test_cpu_end_to_end_six_runs_freeze_test_resume_and_tamper(tmp_path, monkeypatch):
    from tools.emnlp2020_graph_modification_v1.dataset import read, write
    from tools.emnlp2020_task_b_v2 import runner

    torch.set_num_threads(1)
    e = example()
    trainval = {"train": [e], "val": [replace(e, id="val", split="val", group="val-group")]}
    test = replace(e, id="test", split="test", group="test-group")
    texts = [e.instruction, '["café", []]', '["dog", []]', "near"]
    store = TextStore([text_key(t) for t in texts], torch.randn(len(texts), 512))
    dataset = tmp_path / "dataset"
    write(dataset / "trainval-receipt.json", {"fixture": True})
    write(dataset / "test-receipt.json", {"fixture": True})
    write(dataset / "clip-trainval/manifest.json", {"fixture": True})
    config = {
        "dataset": str(dataset),
        "output": str(tmp_path / "run"),
        "weights": "unused",
        "models": ["mlp", "gnn"],
        "seeds": [42, 43, 44],
        "hidden": 8,
        "layers": 1,
        "max_nodes": 2,
        "max_edges": 1,
        "text_batch_size": 2,
        "batch_size": 2,
        "epochs": 1,
        "max_decode_tokens": 16,
        "lr": 0.001,
        "weight_decay": 0.01,
    }
    monkeypatch.setattr(runner, "load_bundle", lambda output, name: {"test": [test]} if name == "test" else trainval)
    monkeypatch.setattr(runner, "frozen_features", lambda *args: store)
    accessed = []

    def prepare_test(*args, **kwargs):
        freeze = read(tmp_path / "run/frozen-checkpoints.json")
        assert len(freeze["runs"]) == 6
        accessed.append("test")
        return {"test": {"usable": 1}}

    monkeypatch.setattr(runner, "prepare", prepare_test)
    freeze = runner.train(config)
    assert not accessed
    assert len(freeze["runs"]) == 6
    assert runner.train(config) == freeze
    result = runner.final_test(config, "unused")
    assert accessed == ["test"]
    assert len(result["runs"]) == 6
    assert result["runs"][0]["metrics"]["examples"] == 1
    assert runner.final_test(config, "unused") == result
    checkpoint = tmp_path / "run/mlp-seed42/best.pt"
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="frozen_checkpoint_changed"):
        runner.final_test(config, "archive-must-not-be-read")
    assert accessed == ["test", "test"]
