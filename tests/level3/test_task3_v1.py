import pytest

from tools.level3_sgg_pretrained_v2.adapter import adapt
from tools.level3_task3_v1.backends import image_seed, resize_source
from tools.level3_task3_v1.core import output_dir, region_ok, validate_tasks
from tools.level3_task3_v1.scoring import align, canonical_source, correlate, iou, score

CFG = {"matching_iou": 0.3, "ambiguity_margin": 0.05, "edit_weight": 0.5, "preservation_weight": 0.5}


def record(objects=None, relations=None, size=(100, 100)):
    objects = (
        objects
        if objects is not None
        else [("person", [0, 0, 20, 50]), ("chair", [25, 30, 50, 70]), ("tree", [70, 0, 100, 90])]
    )
    relations = relations if relations is not None else [(0, "beside", 1), (1, "beside", 2)]
    row = {"image_id": "fixture", "sha256": "test", "width": size[0], "height": size[1]}
    return adapt(
        row,
        "react++",
        [{"label": n, "bbox_xyxy": b, "confidence": 0.9} for n, b in objects],
        [{"subject": s, "object": o, "predicate": p, "confidence": 0.8} for s, p, o in relations],
        {},
        {},
    )


def task(op):
    return {"operation": op}


def test_identity_preservation_does_not_satisfy_removal():
    r = record()
    s = score(r, r, task({"kind": "remove", "source_node": "n001"}), CFG)
    assert s["edit_score"] == 0 and s["preservation_score"] == 1 and s["graph_consistency"] == 0.5


def test_correct_removal_and_unrelated_damage():
    source = record()
    target = record([("person", [0, 0, 20, 50]), ("tree", [70, 0, 100, 90])], [])
    s = score(source, target, task({"kind": "remove", "source_node": "n001"}), CFG)
    assert s["edit_score"] == s["preservation_score"] == s["graph_consistency"] == 1
    target = record([("person", [0, 0, 20, 50])], [])
    s = score(source, target, task({"kind": "remove", "source_node": "n001"}), CFG)
    assert s["edit_score"] == 1 and s["preservation_score"] < 1


def test_deleted_entire_scene_not_perfect_edit():
    s = score(record(), record([], []), task({"kind": "remove", "source_node": "n001"}), CFG)
    assert s["graph_consistency"] == 0.5 and s["preservation_score"] == 0


def test_add_existing_object_never_counts_as_new():
    r = record()
    s = score(r, r, task({"kind": "add", "label": "chair", "region": [0, 0, 1, 1]}), CFG)
    assert s["edit_score"] == 0


def test_add_correct_instance_and_penalize_extra_hallucinations():
    source = record()
    target = record(
        [
            ("person", [0, 0, 20, 50]),
            ("chair", [25, 30, 50, 70]),
            ("tree", [70, 0, 100, 90]),
            ("bicycle", [45, 70, 65, 95]),
        ],
        [(0, "beside", 1), (1, "beside", 2)],
    )
    op = task({"kind": "add", "label": "bicycle", "region": [0.4, 0.6, 0.7, 1]})
    s = score(source, target, op, CFG)
    assert s["graph_consistency"] == 1
    target = record(
        [
            ("person", [0, 0, 20, 50]),
            ("chair", [25, 30, 50, 70]),
            ("tree", [70, 0, 100, 90]),
            ("bicycle", [45, 70, 65, 95]),
            ("dog", [2, 70, 15, 95]),
        ],
        [(0, "beside", 1), (1, "beside", 2)],
    )
    s = score(source, target, op, CFG)
    assert s["edit_score"] == 1 and s["preservation_score"] < 1


def test_add_wrong_location_fails():
    s = score(
        record([], []),
        record([("bird", [0, 0, 10, 10])], []),
        task({"kind": "add", "label": "bird", "region": [0.7, 0.7, 1, 1]}),
        CFG,
    )
    assert s["edit_score"] == 0


def test_replace_needs_absence_and_new_category():
    r = record()
    target = record([("person", [0, 0, 20, 50]), ("potted plant", [25, 30, 50, 70]), ("tree", [70, 0, 100, 90])], [])
    s = score(r, target, task({"kind": "replace", "source_node": "n001", "label": "potted plant"}), CFG)
    assert s["edit_score"] == 1


def test_reversed_relation_not_correct():
    source = record()
    target = record(relations=[(1, "holding", 0), (1, "beside", 2)])
    s = score(
        source,
        target,
        task({"kind": "relation", "subject": "n000", "predicate": "holding", "object": "n001", "present": True}),
        CFG,
    )
    assert s["edit_score"] == 0


def test_negative_relation_requires_both_endpoints():
    source = record()
    target = record([("tree", [70, 0, 100, 90])], [])
    s = score(
        source,
        target,
        task({"kind": "relation", "subject": "n000", "predicate": "beside", "object": "n001", "present": False}),
        CFG,
    )
    assert s["edit_score"] == 0


def test_attribute_and_missing_anchor_abstain():
    r = record()
    s = score(r, r, task({"kind": "attribute", "source_node": "n001", "attribute": "red"}), CFG)
    assert s["graph_consistency"] is None and s["reason"] == "unsupported_attribute_head"
    s = score(r, r, task({"kind": "remove", "source_node": "absent"}), CFG)
    assert s["graph_consistency"] is None


def test_matching_is_one_to_one_and_resolution_normalized():
    source = record([("chair", [0, 0, 30, 30]), ("chair", [0.1, 0.1, 30.1, 30.1])], [])
    target = record([("chair", [0, 0, 60, 60])], [], size=(200, 200))
    matches, ev = align(source, target, CFG)
    assert len(matches) == 1 and len(set(matches.values())) == 1 and ev[0]["ambiguous"]


def test_no_cross_category_alignment():
    a = record([("chair", [0, 0, 30, 30])], [])
    b = record([("dog", [0, 0, 30, 30])], [])
    assert align(a, b, CFG)[0] == {}


def test_resize_control_transfers_original_ids_not_output_order():
    a = record()
    b = record([("tree", [140, 0, 200, 180]), ("person", [0, 0, 40, 100])], [], size=(200, 200))
    canonical, ev = canonical_source(a, b, CFG)
    assert {n["id"]: n["label"] for n in canonical["source_graph"]["nodes"]} == {"n002": "tree", "n000": "person"}
    assert ev["missing_original_anchors"] == ["n001"]


def test_empty_preservation_is_undefined_not_one():
    a = record([], [])
    s = score(a, a, task({"kind": "add", "label": "bird", "region": [0, 0, 1, 1]}), CFG)
    assert s["preservation_score"] is None and s["graph_consistency"] is None


def test_iou():
    assert iou([0, 0, 1, 1], [0, 0, 1, 1]) == 1
    assert iou([0, 0, 0.2, 0.2], [0.8, 0.8, 1, 1]) == 0


def labels(rows):
    return [
        {
            "case_id": r["case_id"],
            "status": "reviewed",
            "reviewer": "fixture",
            "notes": "synthetic CPU test",
            "edit_pass": bool(i),
            "preservation_pass": True,
            "overall_pass": bool(i),
        }
        for i, r in enumerate(rows)
    ]


def test_no_labels_no_fake_correlation():
    r = correlate([], [])
    assert all(v["rho"] is None for v in r["spearman"].values())
    assert not r["ten_disagreements_analyzed"]


def test_common_subset_correlation_and_direction():
    rows = [
        {"case_id": str(i), "graph_consistency": float(i), "clipscore": float(3 - i), "negative_lpips": float(i)}
        for i in range(4)
    ]
    rows[3]["graph_consistency"] = None
    report = correlate(rows, labels(rows))
    assert report["common_subset_n"] == 3
    assert report["spearman"]["graph_consistency"]["rho"] > 0
    assert report["spearman"]["clipscore"]["rho"] < 0
    assert report["disagreement_candidates"] and not report["ten_disagreements_analyzed"]


def test_constant_labels_undefined():
    rows = [{"case_id": str(i), "graph_consistency": i, "clipscore": i, "negative_lpips": i} for i in range(4)]
    reviews = labels(rows)
    for r in reviews:
        r.update(edit_pass=True, overall_pass=True)
    assert correlate(rows, reviews)["spearman"]["graph_consistency"]["rho"] is None


def test_inconsistent_human_labels_rejected():
    rows = [{"case_id": "0", "graph_consistency": 0, "clipscore": 0, "negative_lpips": 0}]
    review = labels(rows)
    review[0]["overall_pass"] = True
    with pytest.raises(ValueError):
        correlate(rows, review)


def test_invalid_regions():
    assert region_ok([0, 0, 1, 1])
    assert not region_ok([0, 0, float("nan"), 1])
    assert not region_ok([0, 0, 2, 1])


def test_unapproved_instruction_pack_rejected():
    r = record()
    p = {
        "schema": "level3-task3-tasks-v1",
        "tasks": [
            {
                "task_id": "test-1",
                "image_id": "fixture",
                "source_sha256": "test",
                "instruction_en": "Remove chair",
                "target_caption_en": "No chair",
                "operation": {"kind": "remove", "source_node": "n001"},
                "status": "draft",
                "reviewer": "",
            }
        ],
    }
    validate_tasks(p, {"fixture": r})
    with pytest.raises(ValueError, match="human_instruction_review_required"):
        validate_tasks(p, {"fixture": r}, approved=True)
    p["tasks"][0].update(status="approved", reviewer="fixture")
    validate_tasks(p, {"fixture": r}, approved=True)


def test_immutable_output_and_dataset_protection(tmp_path):
    from tools.level3_task3_v1 import core

    original = core.RUNS
    try:
        core.RUNS = tmp_path / "runs"
        out = output_dir(core.RUNS / "new")
        with pytest.raises(FileExistsError):
            output_dir(out)
        with pytest.raises(ValueError):
            output_dir(tmp_path / "data")
    finally:
        core.RUNS = original


def test_seed_stable_and_task_specific():
    assert image_seed(42, "a") == image_seed(42, "a")
    assert len({image_seed(42, "a"), image_seed(42, "b"), image_seed(123, "a")}) == 3


def test_resize_geometry(tmp_path):
    from PIL import Image

    p = tmp_path / "source.png"
    Image.new("RGB", (1920, 1080)).save(p)
    assert resize_source(p, 512).size == (512, 288)


def test_moving_existing_object_does_not_count_as_add_or_remove():
    source = record([("chair", [0, 0, 20, 30]), ("tree", [70, 0, 100, 90])], [])
    target = record([("chair", [40, 70, 60, 100]), ("tree", [70, 0, 100, 90])], [])
    assert score(source, target, task({"kind": "remove", "source_node": "n000"}), CFG)["edit_score"] == 0
    assert (
        score(source, target, task({"kind": "add", "label": "chair", "region": [0.3, 0.6, 0.7, 1]}), CFG)["edit_score"]
        == 0
    )


def test_compile_rejects_stale_authoring_before_writing(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from tools.level3_image_graph_v1.core import write
    from tools.level3_task3_v1 import runner

    cfg = {"source_receipt_sha256": "current"}
    monkeypatch.setattr(runner, "sources", lambda: (cfg, {}))
    monkeypatch.setattr(runner, "context", lambda _: {"version": "current"})
    submission = tmp_path / "submission.json"
    write(submission, {"schema": "level3-task3-authoring-v1", "context": {"version": "old"}})
    out = tmp_path / "approved"
    with pytest.raises(ValueError, match="stale_authoring_submission"):
        runner.compile_tasks(SimpleNamespace(submission=submission, output=out))
    assert not out.exists()


def test_report_rejects_smoke_even_with_human_review(tmp_path):
    from types import SimpleNamespace

    from tools.level3_image_graph_v1.core import write
    from tools.level3_task3_v1 import runner
    from tools.level3_task3_v1.core import finish

    run = tmp_path / "smoke"
    run.mkdir()
    write(run / "results.json", [])
    write(run / "run.json", {"evaluation_eligible": False})
    finish(run, "smoke")
    with pytest.raises(ValueError, match="smoke_is_not_benchmark_evidence"):
        runner.report(SimpleNamespace(run=run, review=tmp_path / "unused.json", output=tmp_path / "report"))


def test_report_rejects_tampered_result_before_scoring(tmp_path):
    from types import SimpleNamespace

    from tools.level3_image_graph_v1.core import write
    from tools.level3_task3_v1 import runner
    from tools.level3_task3_v1.core import finish

    run = tmp_path / "pilot"
    run.mkdir()
    write(run / "results.json", [])
    write(run / "run.json", {"evaluation_eligible": True})
    finish(run, "infer")
    (run / "results.json").write_text('[{"tampered": true}]')
    with pytest.raises(ValueError):
        runner.report(SimpleNamespace(run=run, review=tmp_path / "unused.json", output=tmp_path / "report"))
