"""Synthetic CPU-only checks for the actual-review gate, queue, resume and test isolation."""
import json
from dataclasses import replace

import pytest
import torch

from src.editing.operations import EditGraph, Node, Operation, apply_operations
from tools.edit_model_v1.data import Example
from tools.edit_model_v1.text import TextStore, load_store, text_key
from tools.emnlp2020_graph_modification_v1.dataset import read, sha256, write
from tools.magicbrush_experiment_v1 import data, evaluation, pipeline
from tools.magicbrush_experiment_v1.cache import input_texts
from tools.prepare_magicbrush import digest


@pytest.fixture(autouse=True)
def cpu_runtime():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def example(ident, split="train"):
    source = EditGraph((Node("n001", "cup", ("red",)),))
    ops = (Operation("replace_node", source.nodes[0], Node("n001", "cup", ("blue",))),)
    return Example(ident, split, ident, "Synthetic fixture: blue cup", source, apply_operations(source, ops), ops)


def fixture_files(tmp_path):
    rows, labels, states, cases = [], [], {}, []
    for i in range(302):
        split = "train" if i < 300 else "val" if i == 300 else "test"
        e = example(f"synthetic:{i}", split)
        row = {
            "id": e.id,
            "img_id": str(i),
            "turn_index": 1,
            "split": split,
            "source_group_id": e.group,
            "official_split": "train",
            "instruction": e.instruction,
            "source_image": {"sha256": f"s{i}", "pixel_sha256": f"ps{i}"},
            "target_image": {"sha256": f"t{i}", "pixel_sha256": f"pt{i}"},
        }
        label = {
            **{k: row[k] for k in ("id", "split", "source_group_id", "instruction")},
            "source_graph": e.source.to_dict(),
            "target_graph": e.target.to_dict(),
            "operations": [o.to_dict() for o in e.operations],
            "status": "candidate",
            "problems": [],
            "eligible_for_training": split != "test",
            "eligible_for_evaluation": split == "test",
            "annotation_event_sha256": f"event{i}",
        }
        rows.append(row)
        labels.append(label)
        if i < 20:
            states[e.id] = {
                "event_sha256": f"event{i}",
                "source_graph": e.source.to_dict(),
                "target_graph": e.target.to_dict(),
            }
            cases.append(
                {
                    "id": e.id,
                    "decision": "correct",
                    "annotation_event_sha256": f"event{i}",
                    "source_image_sha256": f"s{i}",
                    "target_image_sha256": f"t{i}",
                }
            )
    ids = [c["id"] for c in cases]
    manifest = {
        "records": rows,
        "splits": {s: [r["id"] for r in rows if r["split"] == s] for s in ("train", "val", "test")},
        "review_ids": ids,
    }
    tv = {**manifest, "records": rows[:-1], "splits": {**manifest["splits"], "test": []}}
    test = {
        **manifest,
        "records": rows[-1:],
        "splits": {"train": [], "val": [], "test": [rows[-1]["id"]]},
        "review_ids": [],
    }
    human = {
        "manual_review_completed": True,
        "evidence_source": "user_reported_manual_review_in_conversation",
        "reviewer": "SYNTHETIC TEST USER",
        "user_statements": ["Fixture only, not real review evidence"],
        "review_ids": ids,
        "cases": cases,
        "decision_counts": {"correct": 20, "incorrect": 0, "uncertain": 0},
        "observed_correct_fraction": 1.0,
    }
    payloads = {
        "original_manifest": manifest,
        "trainval_manifest": tv,
        "test_manifest": test,
        "reviewed_states": states,
        "human_review": human,
    }
    paths = {}
    for name, payload in payloads.items():
        paths[name] = tmp_path / f"{name}.json"
        write(paths[name], payload)
    for name, subset, mp in (
        ("trainval_edits", labels[:-1], "trainval_manifest"),
        ("test_edits", labels[-1:], "test_manifest"),
    ):
        paths[name] = tmp_path / f"{name}.json"
        write(paths[name], {"dataset_manifest_sha256": sha256(paths[mp]), "records": subset})
    config = {
        "schema": "magicbrush-manual-experiment-v1",
        "device": "cuda",
        "artifacts": {k: {"path": str(p.relative_to(tmp_path)), "sha256": sha256(p)} for k, p in paths.items()},
        "expected_counts": {"trainval": {"train": 300, "val": 1}, "test": 1},
        "weights": "weights.pt",
        "weights_sha256": "",
        "text_cache": "data/cache",
        "test_text_cache": "data/test-cache",
        "output": "runs/fixture",
        "models": ["mlp", "gnn"],
        "seeds": [42, 123, 3407],
        "text_batch_size": 8,
        "batch_size": 2,
        "val_batch_size": 1,
        "epochs": 1,
        "patience": 1,
        "max_action_tokens": 2048,
        "lr": 0.001,
        "weight_decay": 0.0,
        "gradient_clip": 1.0,
        "model": {"hidden": 8, "byte_dim": 4, "layers": 1, "max_nodes": 20, "max_edges": 60},
    }
    return config, paths


def test_overall_human_verdict_is_supported_without_inventing_axes(tmp_path):
    config, paths = fixture_files(tmp_path)
    examples, report, _ = data.prepare_trainval(config, tmp_path)
    assert {k: len(v) for k, v in examples.items()} == {"train": 300, "val": 1}
    assert report["human_review"]["ready"] and report["human_review"]["observed_correct_fraction"] == 1
    assert not report["human_review"]["per_axis_ratings_invented"]
    assert not report["test_targets_decoded"]
    # Invalid heldout payload remains unread during development preparation.
    paths["test_edits"].write_text('{"test_target": "DO NOT DESERIALIZE DURING TRAINING"}')
    config["artifacts"]["test_edits"]["sha256"] = sha256(paths["test_edits"])
    assert data.prepare_trainval(config, tmp_path)[1]["ready"]


@pytest.mark.parametrize("change", ["event", "image", "graph", "aggregate", "duplicate", "unconfirmed"])
def test_review_gate_rejects_mismatched_or_fabricated_evidence(tmp_path, change):
    _, paths = fixture_files(tmp_path)
    manifest, labels, states, human = (
        read(paths[k]) for k in ("trainval_manifest", "trainval_edits", "reviewed_states", "human_review")
    )
    if change == "event":
        human["cases"][0]["annotation_event_sha256"] = "other"
    if change == "image":
        human["cases"][0]["source_image_sha256"] = "other"
    if change == "graph":
        labels["records"][0]["target_graph"]["nodes"][0]["attributes"] = ["purple"]
    if change == "aggregate":
        human["decision_counts"]["correct"] = 19
    if change == "duplicate":
        human["cases"][0] = human["cases"][1]
    if change == "unconfirmed":
        human["manual_review_completed"] = False
    with pytest.raises(ValueError):
        data.human_gate(manifest, labels, states, human)


def test_input_collection_never_reads_target_or_operations():
    e = example("synthetic:test", "test")
    texts = input_texts([e])
    assert '["cup", ["red"]]' in texts and '["cup", ["blue"]]' not in texts
    e = replace(e, target=object(), operations=object())
    assert input_texts([e]) == texts


def test_split_leakage_and_pin_tampering_rejected(tmp_path):
    config, paths = fixture_files(tmp_path)
    original, tv, test = (read(paths[k]) for k in ("original_manifest", "trainval_manifest", "test_manifest"))
    original["records"][-1]["source_image"]["pixel_sha256"] = original["records"][0]["source_image"]["pixel_sha256"]
    test["records"][0] = original["records"][-1]
    with pytest.raises(ValueError, match="leakage"):
        data.audit_splits(original, tv, test)
    paths["human_review"].write_text("{}")
    with pytest.raises(ValueError, match="Pinned artifact"):
        data.artifact_paths(config, tmp_path)


def test_heldout_requires_explicit_evaluation_only_labels(tmp_path):
    _, paths = fixture_files(tmp_path)
    examples, coverage = data.load_heldout(paths, 1, 2048)
    assert len(examples) == 1 and coverage["evaluated_fraction"] == 1
    labels = read(paths["test_edits"])
    labels["records"][0]["eligible_for_training"] = True
    paths["test_edits"].write_text(json.dumps(labels))
    with pytest.raises(ValueError, match="eligibility"):
        data.load_heldout(paths, 1, 2048)


def test_release_lock_detects_code_and_weight_change(tmp_path, monkeypatch):
    config, _ = fixture_files(tmp_path)
    (tmp_path / "weights.pt").write_bytes(b"CPU TEST WEIGHT PLACEHOLDER")
    config["weights_sha256"] = sha256(tmp_path / "weights.pt")
    cp = tmp_path / "config.json"
    write(cp, config)
    monkeypatch.setattr(pipeline, "source_hashes", lambda: {"fixture.py": "pinned"})
    monkeypatch.setattr(pipeline, "versions", lambda: {"torch": "SYNTHETIC"})
    write(
        cp.with_suffix(".lock.json"),
        {"config_sha256": sha256(cp), "source_sha256": pipeline.source_hashes(), "versions": pipeline.versions()},
    )
    assert pipeline.preflight(cp, root=tmp_path)[2]["ready"]
    monkeypatch.setattr(pipeline, "source_hashes", lambda: {"fixture.py": "changed"})
    with pytest.raises(ValueError, match="Release lock"):
        pipeline.preflight(cp, root=tmp_path)
    monkeypatch.setattr(pipeline, "source_hashes", lambda: {"fixture.py": "pinned"})
    (tmp_path / "weights.pt").write_bytes(b"CHANGED")
    with pytest.raises(ValueError, match="weights"):
        pipeline.preflight(cp, root=tmp_path)


def test_three_seed_statistics_and_unsupported_classes():
    e = example("synthetic")
    from tools.edit_model_v1.codec import encode_actions

    perfect = evaluation.score_rows([e], [{"id": e.id, "tokens": encode_actions(e.operations)[1:], "error": None}])
    zero = evaluation.no_edit([e])
    result = evaluation.aggregate_metrics([perfect, zero, perfect])
    assert result["micro"]["f1"]["mean"] == pytest.approx(2 / 3)
    assert result["micro"]["f1"]["std"] == pytest.approx(3**-0.5)
    assert result["per_type"]["replace_edge"]["f1"]["mean"] is None
    assert result["per_type"]["replace_edge"]["support_per_seed"] == 0


def test_sequential_queue_interrupt_resume_and_test_barrier(tmp_path, monkeypatch):
    config, paths = fixture_files(tmp_path)
    examples, report, _ = data.prepare_trainval(config, tmp_path)
    # Only two synthetic fixtures are learned, never real MagicBrush data or CUDA.
    examples["train"] = examples["train"][:2]
    config["max_action_tokens"] = 4
    identity = {"synthetic_fixture_only": True, "config": config}
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    trained, test_reads, fail = [], [], [True]
    real_train, real_eval, real_test = pipeline.train_one, pipeline.evaluate_one, pipeline.load_heldout

    def cache(es, cfg, ident, directory, root, *, scope):
        if (directory / "manifest.json").exists():
            return load_store(directory, ident)
        texts = input_texts(es)
        vectors = torch.randn(len(texts), 512, generator=torch.Generator().manual_seed(2))
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
        return TextStore(meta["text_sha256"], vectors), meta

    def train(es, store, cfg, ident, kind, seed, directory, **kwargs):
        if len(trained) == 2 and fail[0]:
            fail[0] = False
            raise KeyboardInterrupt("simulated user interruption")
        trained.append((kind, seed))
        return real_train(es, store, cfg, ident, kind, seed, directory, device="cpu", wandb_mode="disabled")

    def heldout(ps, count, budget):
        output = tmp_path / config["output"]
        assert len(list(output.glob("*/*/completed-receipt.json"))) == 6
        assert (output / "selection.json").exists()
        test_reads.append(True)
        return real_test(ps, count, 2048)

    def evaluate(*args, **kwargs):
        return real_eval(*args, **kwargs, device="cpu")

    monkeypatch.setattr(pipeline, "build_cache", cache)
    monkeypatch.setattr(pipeline, "train_one", train)
    monkeypatch.setattr(pipeline, "load_heldout", heldout)
    monkeypatch.setattr(pipeline, "evaluate_one", evaluate)
    with pytest.raises(KeyboardInterrupt):
        pipeline.execute(config, examples, report, identity, paths, wandb_mode="disabled")
    assert len(trained) == 2 and test_reads == []
    output = tmp_path / config["output"]
    training_sha = digest(read(output / "experiment.json"))
    with pytest.raises(FileNotFoundError):
        evaluation.freeze_selection(output, config, training_sha)
    pipeline.execute(config, examples, report, identity, paths, wandb_mode="disabled")
    assert trained == [(k, s) for k in config["models"] for s in config["seeds"]]
    result = read(output / "summary.json")
    assert result["models"]["gnn"]["test"]["aggregate"]["micro"]["f1"]["n_total"] == 3
    assert read(output / "progress.json")["stage"] == "complete"
    checksum = sha256(output / "summary.json")
    pipeline.execute(config, examples, report, identity, paths, wandb_mode="disabled")
    assert len(trained) == 6 and sha256(output / "summary.json") == checksum
    # Resume refuses changed completed predictions rather than silently rerunning them.
    (output / "mlp/42/predictions-val.json").write_text("[]")
    with pytest.raises(ValueError, match="modified"):
        pipeline.execute(config, examples, report, identity, paths, wandb_mode="disabled")
