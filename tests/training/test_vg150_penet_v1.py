"""Synthetic CPU contracts for the new VG150 study, never real training evidence."""

import copy
import json

import pytest
import torch

from src.datasets.schemas import SceneGraphRecord, SceneObject, SceneRelation
from src.evaluation.reference_predcls import evaluate_reference_image, summarize_reference
from src.graph.builder import SceneGraphBuilder
from src.training import gqa_full
from src.training.gqa_pilot import collate, soft_targets
from tools.prepare_magicbrush import digest
from tools.run_gqa_full_prepared import prepare_supervision, prepared_collate
from tools.run_gqa_kg_pilot import assert_exact
from tools.vg150_penet_v1 import data
from tools.vg150_penet_v1 import experiment as study
from tools.vg150_penet_v1.prepare import MODELS, train_statistics, validate


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("VG150 CPU tests must not query CUDA")

    # AdamW performs a harmless availability check even for CPU tensors.
    # Hide CUDA and forbid actual initialization/allocation in these fixtures.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


def record(identifier="synthetic"):
    return SceneGraphRecord(
        identifier,
        100,
        100,
        (SceneObject(1, 1, (0, 0, 20, 20)), SceneObject(2, 150, (30, 0, 50, 20)), SceneObject(3, 3, (60, 0, 80, 20))),
        (SceneRelation(1, 2, 1), SceneRelation(1, 2, 50), SceneRelation(1, 2, 1), SceneRelation(2, 3, 2)),
    )


def fixtures():
    torch.manual_seed(9)
    graph = prepare_supervision(SceneGraphBuilder().build(record(), torch.randn(3, 517)), 51)
    empty = prepare_supervision(SceneGraphBuilder().build(SceneGraphRecord("empty", 100, 100), torch.empty(0, 517)), 51)
    vectors = [torch.randn(151, 300), torch.randn(51, 300)]
    features = torch.randn(151, 4)
    features[0] = 0
    valid = torch.ones(151, dtype=torch.bool)
    valid[0] = False
    config = {
        "scope": "synthetic_cpu_fixture",
        "device": "cpu",
        "epochs": 2,
        "lr": 0.001,
        "weight_decay": 0.01,
        "train_batch_size": 1,
        "val_batch_size": 1,
        "gradient_clip_norm": 5.0,
        "checkpoint_every_batches": 1,
        "early_stopping": {"enabled": True, "patience": 15, "min_delta": 0.0},
        "wandb": {"mode": "disabled"},
    }
    dimensions = {
        "geometry_dim": 517,
        "embedding_dim": 4,
        "hidden_dim": 8,
        "num_classes": 51,
        "num_object_classes": 151,
        "semantic_dim": 8,
        "pe_dropout": 0.2,
        "tail_ids": list(range(11, 51)),
    }
    return [graph, empty, graph], [graph], vectors, features, valid, {"config": config, "dimensions": dimensions}


def test_cohorts_are_stable_disjoint_and_reuse_full_row_order():
    splits = {"train": list(range(20)), "val": list(range(20, 30)), "test": list(range(30, 35))}
    assert data.cohort(splits) == {k: splits[k] for k in ("train", "val")}
    a = data.cohort(splits, pilot=True, train_size=4, val_size=2)
    reversed_splits = {k: list(reversed(v)) for k, v in splits.items()}
    assert a == data.cohort(reversed_splits, pilot=True, train_size=4, val_size=2)
    with pytest.raises(ValueError, match="Non-disjoint"):
        data.cohort({**splits, "test": [0]})


def test_multi_predicate_pairs_and_duplicate_GT_are_not_silently_dropped():
    graph = fixtures()[0][0]
    target = soft_targets(graph, 51)
    pair = graph.edge_label_index.T.tolist().index([0, 1])
    assert graph.edge_label[pair] == -1 and target[pair, 1] == target[pair, 50] == 0.5
    assert graph.gt_predicate.tolist() == [1, 50, 1, 2]
    for generator in (None, torch.Generator().manual_seed(42)):
        second = None if generator is None else torch.Generator().manual_seed(42)
        a, slices = collate([graph], 51, generator)
        b, other = prepared_collate([graph], 51, second)
        assert_exact(a, b)
        assert slices == other


def test_self_relations_excluded_explicitly_but_other_duplicates_retained():
    original = record()
    original = SceneGraphRecord(
        original.image_id, 100, 100, original.objects, original.relations + (SceneRelation(1, 1, 3),)
    )
    assert data.normalized_record(original).relations == record().relations


def test_topology_has_no_GT_dependency():
    original = record()
    changed = SceneGraphRecord(original.image_id, 100, 100, original.objects, (SceneRelation(3, 1, 7),))
    features = torch.randn(3, 517)
    a, b = [SceneGraphBuilder().build(r, features) for r in (original, changed)]
    assert torch.equal(a.edge_index, b.edge_index) and torch.equal(a.x, b.x)


def test_statistics_reject_validation_and_count_train_duplicate_policy():
    class Loader:
        split = "val"

        def __len__(self):
            return 1

        def load(self, index):
            return record()

    loader = Loader()
    with pytest.raises(ValueError, match="only inspect training"):
        train_statistics(loader)
    loader.split = "train"
    stats = train_statistics(loader)
    assert stats["GT_rows"] == 4 and stats["unique_GT_triplets"] == 3 and stats["multi_predicate_pairs"] == 1
    assert stats["object_support"]["150"] == 1 and stats["predicate_support"]["50"] == 1


def test_dataset_rejects_test_and_out_of_split_rows():
    class Loader:
        split = "test"
        indices = [0]

    with pytest.raises(ValueError, match="train/val"):
        data.Dataset(Loader(), None, [0])
    loader = Loader()
    loader.split = "train"
    with pytest.raises(ValueError, match="train/val"):
        data.Dataset(loader, None, [1])


def test_private_trainer_does_not_modify_frozen_global_functions():
    values = fixtures()
    before = (gqa_full.create_model, gqa_full.forward, gqa_full.evaluate, gqa_full.collate, gqa_full.RunTracker)
    trainer = study.isolated_trainer(values[3], values[4])
    assert trainer._train_run.__globals__ is not gqa_full._train_run.__globals__
    assert before == (gqa_full.create_model, gqa_full.forward, gqa_full.evaluate, gqa_full.collate, gqa_full.RunTracker)


@pytest.mark.parametrize("kind", MODELS)
def test_cpu_training_exact_resume_full_metrics_and_checkpoint_integrity(tmp_path, kind):
    train, val, vectors, features, valid, experiment = fixtures()
    trainer = study.isolated_trainer(features, valid)
    expected = trainer.train_run(kind, 42, train, val, vectors, experiment, tmp_path / "continuous")
    assert (
        trainer.train_run(kind, 42, train, val, vectors, experiment, tmp_path / "resumed", stop_after_batches=1) is None
    )
    torch.manual_seed(999)
    actual = trainer.train_run(kind, 42, train, val, vectors, experiment, tmp_path / "resumed")
    for key in ("history", "validation", "initial_state_sha256", "selected_epoch"):
        assert expected[key] == actual[key]
    a, b = [
        torch.load(tmp_path / d / "last.pt", map_location="cpu", weights_only=True) for d in ("continuous", "resumed")
    ]
    for key in ("model", "optimizer", "sampler_rng", "torch_rng", "best_model", "total_batches"):
        assert_exact(a[key], b[key])
    assert actual["validation"]["gc"]["num_predicates_for_mean"] == 50
    assert actual["validation"]["gc"]["per_predicate"]["1"]["support"] == 2
    assert "VG150" in actual["validation"]["gc"]["protocol"] and not actual["test_used"]
    assert "tail40_mR@50" in actual["validation"]["gc"]
    if kind == "gat-kg":
        for key in ("kg_features", "kg_valid"):
            assert key in a["model"]
    assert trainer.train_run(kind, 42, train, val, vectors, experiment, tmp_path / "resumed") == actual
    changed = copy.deepcopy(experiment)
    changed["config"]["lr"] *= 2
    with pytest.raises(ValueError, match="Pinned identity"):
        trainer.train_run(kind, 42, train, val, vectors, changed, tmp_path / "resumed")


def test_pe_modes_initial_state_and_all_sampling_traces_match(tmp_path):
    train, val, vectors, features, valid, experiment = fixtures()
    experiment["config"]["epochs"] = 1
    results = {
        kind: study.isolated_trainer(features, valid).train_run(
            kind, 42, train, val, vectors, experiment, tmp_path / kind
        )
        for kind in MODELS
    }
    assert len({r["initial_state_sha256"] for k, r in results.items() if k.startswith("pe-")}) == 1
    assert len({tuple(h["sampling_trace_sha256"] for h in r["history"]) for r in results.values()}) == 1


def test_reference_evaluation_matches_direct_frozen_matcher():
    train, val, vectors, features, valid, experiment = fixtures()
    model = study.create_model("gat", experiment["dimensions"], vectors, features, valid).eval()
    graph = val[0]
    with torch.inference_mode():
        logits = model(graph.x, graph.edge_index, graph.edge_label_index, graph.object_labels)
    expected = summarize_reference(
        [
            evaluate_reference_image(
                logits,
                graph.edge_label_index,
                graph.gt_relation_index,
                graph.gt_predicate,
                graph.boxes,
                graph.object_labels,
            )
        ]
    )
    actual = study.evaluate(model, "gat", val, {**experiment["dimensions"], "batch_size": 1}, "cpu")["gc"]
    for key in ("R@50", "R@100", "mR@50", "mR@100", "per_predicate"):
        assert expected[key] == actual[key]


def test_main_training_requires_six_verified_pilots(tmp_path, monkeypatch):
    config = {"output": str(tmp_path)}
    monkeypatch.setattr(study, "experiment", lambda *a, **k: {"fixture": True})
    with pytest.raises(ValueError, match="needs completed matched pilot"):
        study.require_pilots(config, {})


def test_config_and_frozen_source_fingerprint_are_CPU_only():
    config = json.loads(study.DEFAULT_CONFIG.read_text())
    validate(config)
    hashes = study.source_hashes(config)
    assert "src/models/penet.py" in hashes and "tools/vg150_penet_v1/experiment.py" in hashes
    assert len(digest(config)) == 64
