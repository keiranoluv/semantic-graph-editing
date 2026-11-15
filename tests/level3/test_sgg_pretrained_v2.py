import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tools.level3_image_graph_v1.core import sha256, write
from tools.level3_sgg_pretrained_v2.adapter import adapt, react_decode
from tools.level3_sgg_pretrained_v2.comparison import export, make_comparison
from tools.level3_sgg_pretrained_v2.evaluation import evaluate, template
from tools.level3_sgg_pretrained_v2.runner import CONFIG, new_output, verified_bundle


def row(i=0):
    return {"image_id": f"image-{i}", "sha256": f"hash-{i}", "width": 200, "height": 100, "image_path": f"{i}.png"}


def record(r=None):
    r = r or row()
    return adapt(
        r,
        "react++",
        [
            {"label": "person", "bbox_xyxy": [0, 0, 20, 40], "confidence": 0.8},
            {"label": "bicycle", "bbox_xyxy": [15, 20, 50, 60], "confidence": 0.9},
        ],
        [{"subject": 0, "object": 1, "predicate": "riding", "confidence": 0.7}],
        {},
        {},
    )


def comparison():
    rows = [row(i) for i in range(5)]
    return make_comparison(rows, {"react++": [record(r) for r in rows]})


def reviewed():
    c = comparison()
    review = template(c)
    r = next(r for r in review["reviews"] if r["model"] == "react++")
    r.update(
        status="reviewed",
        reviewer="tester",
        notes="synthetic CPU test only",
        inventory_complete=True,
        missing_objects=["missed-person"],
        missing_relations=["missed-relation"],
    )
    for i, obj in enumerate(r["objects"].values()):
        obj.update(correct=True, grounded=True, hallucinated=False, reference_instance=f"gt-{i}")
    for obj in r["relations"].values():
        obj.update(correct=True, hallucinated=False, reference_relation="gt-edge")
    return c, review, r


def test_react_letterbox_direction_and_low_confidence_endpoint():
    boxes = np.array([[0, 160, 64, 288, 1, 0.9], [48, 224, 160, 352, 2, 0.1]])
    rels = np.array([[0, 1, 1, 0.06, 0.8]])
    objects, relations, evidence = react_decode(
        boxes,
        rels,
        {1: "person", 2: "bicycle"},
        {1: "riding"},
        3.2,
        (0, 160),
        200,
        100,
        {"box_threshold": 0.4, "relation_threshold": 0.05},
    )
    assert objects[0]["bbox_xyxy"] == [0, 0, 20, 40]
    assert len(objects) == 2 and relations[0]["subject"] == 0 and relations[0]["object"] == 1
    assert evidence["selected_native_indices"] == [0, 1]
    assert boxes[0, 1] == 160  # original native tensor untouched


@pytest.mark.parametrize("bad", [np.nan, 0.5, -1, 5])
def test_react_bad_endpoint_rejected_even_below_threshold(bad):
    with pytest.raises(ValueError):
        react_decode(
            np.array([[0, 0, 10, 10, 1, 0.8]]),
            np.array([[bad, 0, 1, 0, 0.1]]),
            {1: "person"},
            {1: "on"},
            1,
            (0, 0),
            100,
            100,
            {"box_threshold": 0.4, "relation_threshold": 0.05},
        )


def test_duplicate_relations_preserve_native_evidence():
    r = record()
    native = [{"subject": 0, "object": 1, "predicate": "on", "confidence": 0.7}] * 2
    out = adapt(
        row(),
        "x",
        [
            dict(v, label=n["label"])
            for n, v in zip(r["source_graph"]["nodes"], r["grounding"]["nodes"].values(), strict=False)
        ],
        native,
        {},
        {},
    )
    assert len(out["source_graph"]["edges"]) == 1
    assert out["grounding"]["edges"]["e000"]["native_rows"] == [0, 1]
    assert all(not n["attributes"] for n in out["source_graph"]["nodes"])


@pytest.mark.parametrize("score", [float("nan"), -1, 1.1])
def test_invalid_confidence(score):
    with pytest.raises(ValueError):
        adapt(row(), "x", [{"label": "x", "bbox_xyxy": [0, 0, 20, 20], "confidence": score}], [], {}, {})


def test_pending_metrics_are_null_not_zero():
    metrics = evaluate(comparison())["models"]
    assert metrics["react++"]["parse_success"] == 1
    assert metrics["react++"]["objects"] is None
    assert metrics["react++"]["hallucination"]["objects"]["rate"] is None
    assert metrics["robin-sam"]["parse_success"] is None


def test_human_counts_and_missing_inventory():
    c, review, _ = reviewed()
    m = evaluate(c, review)["models"]["react++"]
    assert m["objects"]["precision"] == 1
    assert m["objects"]["recall"] == pytest.approx(2 / 3)
    assert m["objects"]["f1"] == 0.8
    assert m["relations"]["f1"] == pytest.approx(2 / 3)
    assert m["complete_inventory_images"] == 1


@pytest.mark.parametrize(
    "mutation",
    ["stale", "anonymous", "duplicate_ref", "uncertain", "bad_endpoint", "missing_overlap", "missing_coverage"],
)
def test_invalid_reviews(mutation):
    c, review, r = reviewed()
    if mutation == "stale":
        review["comparison_sha256"] = "bad"
    if mutation == "anonymous":
        r["reviewer"] = ""
    if mutation == "duplicate_ref":
        r["objects"]["n001"]["reference_instance"] = "gt-0"
    if mutation == "uncertain":
        r["objects"]["n000"]["correct"] = None
    if mutation == "bad_endpoint":
        r["objects"]["n000"]["correct"] = False
    if mutation == "missing_overlap":
        r["missing_objects"] = ["gt-0"]
    if mutation == "missing_coverage":
        del r["objects"]["n000"]
    with pytest.raises(ValueError):
        evaluate(c, review)


def test_partial_review_does_not_generate_recall():
    c, review, r = reviewed()
    r["inventory_complete"] = False
    assert evaluate(c, review)["models"]["react++"]["objects"] is None


def test_failures_count_in_parse_denominator():
    c = comparison()
    c["cases"][0]["models"]["react++"]["record"]["status"] = "parse_error"
    assert evaluate(c)["models"]["react++"]["parse_success"] == 0.8


def test_cohort_rejects_hash_mismatch_and_duplicates():
    rows = [row(i) for i in range(5)]
    bad = record()
    bad["source_sha256"] = "bad"
    with pytest.raises(ValueError):
        make_comparison(rows, {"react++": [bad]})
    with pytest.raises(ValueError):
        make_comparison(rows, {"react++": [record(), record()]})


def test_receipt_detects_tampering(tmp_path):
    write(tmp_path / "records.json", [])
    write(
        tmp_path / "receipt.json",
        {"execution_complete": True, "files_sha256": {"records.json": sha256(tmp_path / "records.json")}},
    )
    assert verified_bundle(tmp_path, "records.json") == []
    (tmp_path / "records.json").write_text("[1]")
    with pytest.raises(ValueError):
        verified_bundle(tmp_path, "records.json")


def test_immutable_output(tmp_path):
    cfg = json.loads(CONFIG.read_text())
    with pytest.raises(FileExistsError):
        new_output(tmp_path, cfg)
    with pytest.raises(ValueError):
        new_output(Path("data/processed/forbidden"), cfg)


def test_self_contained_three_model_viewer(tmp_path):
    locked = tmp_path / "locked"
    locked.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    rows = []
    for i in range(5):
        r = row(i)
        Image.new("RGB", (200, 100)).save(locked / r["image_path"])
        r["sha256"] = sha256(locked / r["image_path"])
        rows.append(r)
    export(rows, {"react++": [record(r) for r in rows]}, locked, output)
    text = (output / "comparison.html").read_text()
    assert "repeat(3,minmax(0,1fr))" in text and "data:image/jpeg;base64," in text
    assert "Directed graph" in text and "relate-anything" in text and "not_run" in text
    assert json.loads((output / "metrics.json").read_text())["models"]["react++"]["objects"] is None


def test_same_image_requires_shared_human_reference_inventory():
    from copy import deepcopy

    c = comparison()
    for case in c["cases"]:
        case["models"]["relate-anything"] = deepcopy(case["models"]["react++"])
        case["models"]["relate-anything"]["record"]["model"] = "relate-anything"
    review = template(c)
    for row in review["reviews"]:
        if row["candidate_id"] != "image-0" or row["model"] == "robin-sam":
            continue
        row.update(status="reviewed", reviewer="tester", notes="CPU fixture", inventory_complete=True)
        row["missing_objects"] = ["extra"] if row["model"] == "react++" else []
        for i, obj in enumerate(row["objects"].values()):
            obj.update(correct=True, grounded=True, hallucinated=False, reference_instance=f"gt-{i}")
        for rel in row["relations"].values():
            rel.update(correct=True, hallucinated=False, reference_relation="gt-edge")
    with pytest.raises(ValueError, match="share_one_human_reference_inventory"):
        evaluate(c, review)
