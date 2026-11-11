"""Audit regression tests: annotation-row counts, nested IDs and held-out isolation."""

import json
from collections import Counter

import pytest

from src.datasets.schemas import SceneGraphRecord, SceneObject, SceneRelation
from src.knowledge.transe.core import sha256
from tools.audit_gqa_scale import describe_record, main, ordered_train_ids, summarize


def test_audit_distinguishes_rows_triplets_pairs_and_empty_graphs():
    objects = tuple(SceneObject(i, 1, (i * 10.0, 0.0, i * 10.0 + 5.0, 5.0)) for i in range(3))
    record = SceneGraphRecord(
        "a", 100, 100, objects, (SceneRelation(0, 1, 1), SceneRelation(0, 1, 1), SceneRelation(0, 1, 2))
    )
    first = describe_record(record, Counter(images=1, relations=3))
    assert (first["gt_annotation_rows"], first["unique_gt_triplets"], first["positive_pairs"]) == (3, 2, 1)
    assert first["multi_predicate_pairs"] == 1
    assert first["candidate_pairs"] == 6 and first["sampled_training_pairs"] == 4
    singleton = describe_record(SceneGraphRecord("b", 100, 100, objects[:1]), Counter(images=1))
    background = describe_record(SceneGraphRecord("c", 100, 100, objects[:2]), Counter(images=1))
    stats, rows, images = summarize([first, singleton, background], 3, [1])
    assert rows == {1: 2, 2: 1} and images == {1: 1, 2: 1}
    assert stats["images_without_gt"] == 2 and stats["images_without_candidates"] == 1
    assert stats["sampled_training_pairs"] == 6
    assert stats["directional_fraction"] == pytest.approx(2 / 3)
    assert stats["predicate_coverage"]["absent_ids"] == [3]
    assert stats["objects"]["p50"] == 2
    assert stats["candidates"]["total"] == 8
    assert stats["cost_components"]["largest_image_fp32_all_pair_logits_bytes"] == 6 * 4 * 4


def test_train_cohort_prefixes_ignore_input_order():
    ids = [str(i) for i in range(100)]
    ordered = ordered_train_ids(ids)
    assert ordered_train_ids(reversed(ids)) == ordered
    assert set(ordered[:10]) < set(ordered[:20]) < set(ordered)
    assert ordered != ordered_train_ids(ids, seed=123)


def test_audit_cli_never_opens_test_annotations_and_resumes_immutably(tmp_path, monkeypatch):
    scene = tmp_path / "sceneGraphs"
    scene.mkdir()
    annotations = {image_id: {"width": 100, "height": 100, "objects": {}} for image_id in ("a", "b", "c")}
    train_path = scene / "train_sceneGraphs.json"
    train_path.write_text(json.dumps(annotations))
    test_path = scene / "val_sceneGraphs.json"
    test_path.write_text("NOT VALID JSON: must not be opened")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "splits": {"train": ["a", "b"], "val": ["c"], "test": ["held-out"]},
                "annotation_sha256": {"train_sceneGraphs.json": sha256(train_path), "val_sceneGraphs.json": "not-read"},
                "vocabulary": {
                    "label_to_idx": {"thing": 1},
                    "attribute_to_idx": {},
                    "predicate_to_idx": {"to the left of": 1, "to the right of": 2},
                },
            }
        )
    )
    cohort = tmp_path / "cohort.json"
    cohort.write_text(
        json.dumps(
            {
                "manifest_sha256": sha256(manifest),
                "seed": 42,
                "splits": {"train": ordered_train_ids(["a", "b"])[:1], "val": ["c"]},
            }
        )
    )
    report = tmp_path / "report.json"
    from pathlib import Path

    original = Path.open

    def guarded_open(path, *args, **kwargs):
        if path == test_path:
            pytest.fail("Audit opened held-out test annotation")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    args = [
        "--root",
        str(tmp_path),
        "--manifest",
        str(manifest),
        "--pilot-cohort",
        str(cohort),
        "--report",
        str(report),
    ]
    main(args)
    digest = sha256(report)
    main(args)
    assert sha256(report) == digest
    data = json.loads(report.read_text())
    assert data["test_annotation_opened"] is False
    assert data["summaries"]["train_full"]["images"] == 2
    assert data["summaries"]["val_full"]["images"] == 1
