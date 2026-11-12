"""CPU evidence-policy tests. Synthetic fixtures never count as dataset labels."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from src.editing.operations import EditGraph, Operation, apply_operations
from tools.magicbrush_grounded_v1.language import caption_nouns, scope
from tools.magicbrush_grounded_v1.models import decode_detections, query_text
from tools.magicbrush_grounded_v1.policy import analyze
from tools.magicbrush_grounded_v1.runner import deserialize_evidence, serialize_evidence

CONFIG = json.loads(
    (Path(__file__).resolve().parents[2] / "configs/level2/magicbrush-grounded-anchor-v1.json").read_text()
)


def node(identifier, label, box, color=None):
    return {
        "id": identifier,
        "label": label,
        "bbox": box,
        "embedding": [1.0, 0.0],
        "color_scores": {color: 0.4, "gray" if color != "gray" else "red": 0.2} if color else {},
        "color_fractions": {color: 0.9} if color else {},
    }


def proposal(nodes, scores=None):
    return {"nodes": nodes, "presence_scores": scores or {}, "overflow": False, "ambiguous_focus": False}


def proof(*, cosine=1.0, mae=0.0, old=0.4, new=0.4):
    return {"cosine": cosine, "mae": mae, "source_class_score": old, "target_class_score": new}


def evidence(source, target, boxes):
    mask = np.zeros((100, 100), dtype=bool)
    for a, b, c, d in boxes:
        mask[int(b / 10) : int(d / 10), int(a / 10) : int(c / 10)] = True
    return {
        "source_anchors": {n["id"]: proof() for n in source["nodes"]},
        "target_anchors": {n["id"]: proof() for n in target["nodes"]},
        "change_mask": mask,
        "changed_pixels": int(mask.sum()),
        "comparable_aspect_ratio": True,
    }


def verify_transaction(result):
    assert apply_operations(
        EditGraph.from_dict(result["source_graph"]), [Operation.from_dict(o) for o in result["operations"]]
    ) == EditGraph.from_dict(result["target_graph"])
    assert result["eligible_for_training"] is False
    assert result["human_review_status"] == "pending"
    assert result["automatic_approval"] is False


@pytest.mark.parametrize(
    "instruction",
    [
        "make the man hold a wine bottle",
        "remove the computer and add a stack of papers",
        "make it blue",
        "Add a second airplane",
        "make the image a cartoon",
        "add a big rubber duck on the hydrant",
    ],
)
def test_unverified_semantics_are_explicitly_unsupported(instruction):
    assert scope(instruction)["supported"] is False


def test_typed_open_vocabulary_scope_and_source_caption_nouns():
    assert scope("make the entire hydrant blue")["after"] == {"label": "hydrant", "color": "blue"}
    assert scope("Replace the mirror with a window.")["kind"] == "replace"
    assert scope("Add a pizza cutter.")["after"]["label"] == "pizza cutter"
    assert scope("move the cup to the right of the table")["direction"] == "right_of"
    assert "cat" in caption_nouns("a cat sitting on a table")


def test_color_edit_preserves_identity_context_edges_and_source_geometry():
    source = proposal([node("s1", "shirt", [100, 100, 300, 400], "red"), node("s2", "chair", [600, 100, 900, 800])])
    target = proposal([node("t8", "shirt", [105, 100, 305, 400], "blue"), node("t3", "chair", [600, 100, 900, 800])])
    ev = evidence(source, target, [[100, 100, 305, 400]])
    ev["source_anchors"]["s1"] = proof(cosine=0.95, mae=0.2)
    result = analyze(source, target, ev, "make the shirt blue", CONFIG)
    assert result["status"] == "candidate"
    assert [op["kind"] for op in result["operations"]] == ["replace_node"]
    assert result["source_graph"]["edges"] == result["target_graph"]["edges"]
    assert result["source_boxes"] == result["target_boxes"]
    assert result["operations"][0]["after"]["id"] == "n001"
    verify_transaction(result)


def test_detector_absence_is_not_sufficient_to_delete():
    source = proposal([node("old", "cup", [100, 100, 300, 400]), node("ctx", "table", [600, 100, 900, 800])])
    target = proposal([node("ctx2", "table", [600, 100, 900, 800])], {"cup": 0.01})
    ev = evidence(source, target, [[100, 100, 300, 400]])
    result = analyze(source, target, ev, "remove the cup", CONFIG)
    assert result["status"] == "needs_review"
    assert "deletion_has_insufficient_visual_evidence" in result["problems"]
    assert not result["operations"]
    ev["source_anchors"]["old"] = proof(cosine=0.6, mae=0.2, old=0.4, new=0.1)
    result = analyze(source, target, ev, "remove the cup", CONFIG)
    assert result["status"] == "candidate"
    assert {op["kind"] for op in result["operations"]} == {"delete_node", "delete_edge"}
    verify_transaction(result)


def test_addition_requires_class_contrast_and_locality():
    source = proposal([node("ctx", "table", [600, 100, 900, 800])], {"cup": 0.01})
    target = proposal([node("ctx2", "table", [600, 100, 900, 800]), node("new", "cup", [100, 100, 300, 400])])
    ev = evidence(source, target, [[100, 100, 300, 400]])
    ev["target_anchors"]["new"] = proof(cosine=0.6, mae=0.2, old=0.1, new=0.4)
    result = analyze(source, target, ev, "add a cup", CONFIG)
    assert result["status"] == "candidate"
    assert {op["kind"] for op in result["operations"]} == {"add_node", "add_edge"}
    verify_transaction(result)
    ev["target_anchors"]["new"]["source_class_score"] = 0.4
    assert "addition_has_insufficient_visual_evidence" in analyze(source, target, ev, "add a cup", CONFIG)["problems"]
    ev["target_anchors"]["new"]["source_class_score"] = 0.1
    ev["change_mask"][:] = True
    ev["changed_pixels"] = 10000
    assert "image_changes_outside_verified_focus" in analyze(source, target, ev, "add a cup", CONFIG)["problems"]


def test_replacement_keeps_existing_instance_id():
    source = proposal(
        [node("old", "mirror", [100, 100, 300, 400]), node("ctx", "table", [600, 100, 900, 800])], {"window": 0.01}
    )
    target = proposal(
        [node("new", "window", [100, 100, 300, 400]), node("ctx2", "table", [600, 100, 900, 800])], {"mirror": 0.01}
    )
    ev = evidence(source, target, [[100, 100, 300, 400]])
    ev["source_anchors"]["old"] = proof(cosine=0.6, mae=0.2, old=0.4, new=0.1)
    ev["target_anchors"]["new"] = proof(cosine=0.6, mae=0.2, old=0.1, new=0.4)
    result = analyze(source, target, ev, "replace the mirror with a window", CONFIG)
    assert result["status"] == "candidate"
    assert [op["kind"] for op in result["operations"]] == ["replace_node"]
    verify_transaction(result)


def test_move_changes_relation_without_delete_add_of_stable_instance():
    source = proposal([node("old", "cup", [100, 100, 250, 300]), node("ctx", "table", [400, 100, 600, 300])])
    target = proposal([node("new", "cup", [700, 100, 850, 300]), node("ctx2", "table", [400, 100, 600, 300])])
    ev = evidence(source, target, [[100, 100, 250, 300], [700, 100, 850, 300]])
    ev["source_anchors"]["old"] = proof(cosine=0.6, mae=0.2)
    result = analyze(source, target, ev, "move the cup to the right of the table", CONFIG)
    assert result["status"] == "candidate"
    assert [op["kind"] for op in result["operations"]] == ["replace_edge"]
    assert result["source_graph"]["nodes"] == result["target_graph"]["nodes"]
    verify_transaction(result)


def test_context_change_and_duplicate_instances_force_abstention():
    source = proposal([node("old", "cup", [100, 100, 300, 400]), node("ctx", "table", [600, 100, 900, 800])])
    target = proposal([node("ctx2", "table", [600, 100, 900, 800])], {"cup": 0.01})
    ev = evidence(source, target, [[100, 100, 300, 400]])
    ev["source_anchors"]["old"] = proof(cosine=0.6, mae=0.2, old=0.4, new=0.1)
    ev["source_anchors"]["ctx"] = proof(cosine=0.8, mae=0.1)
    result = analyze(source, target, ev, "remove the cup", CONFIG)
    assert "source_context_not_visually_stable" in result["problems"]
    source["nodes"].append(node("old2", "cup", [100, 600, 300, 900]))
    ev["source_anchors"]["old2"] = proof()
    assert "source_focus_reference_not_unique" in analyze(source, target, ev, "remove the cup", CONFIG)["problems"]


def test_query_span_scoring_keeps_low_score_absence_evidence_before_thresholding():
    queries = ["duck", "fire hydrant"]
    text, spans = query_text(queries)
    assert text == "duck. fire hydrant."
    offsets = [(0, 0), (0, 4), (4, 5), (6, 10), (11, 18), (18, 19), (0, 0)]
    logits = np.full((2, 7), -9.0)
    logits[0, 1] = 3
    logits[1, 3:5] = -3
    logits[:, -1] = -np.inf  # Real DINO pads unused text positions with -inf.
    result = decode_detections(
        logits, np.array([[0.2, 0.2, 0.2, 0.2], [0.7, 0.7, 0.2, 0.2]]), offsets, spans, queries, queries, CONFIG
    )
    assert [n["label"] for n in result["nodes"]] == ["duck"]
    assert 0 < result["presence_scores"]["fire hydrant"] < 0.15
    with pytest.raises(ValueError, match="missing or truncated"):
        decode_detections(
            logits, np.array([[0.2, 0.2, 0.2, 0.2], [0.7, 0.7, 0.2, 0.2]]), offsets[:4], spans, queries, queries, CONFIG
        )


def test_mask_serialization_is_exact_and_detects_corruption():
    mask = np.array([[True, False, False], [False, True, True]])
    record = {"change_mask": mask, "changed_pixels": 3}
    encoded = serialize_evidence(record)
    assert np.array_equal(deserialize_evidence(encoded)["change_mask"], mask)
    bad = copy.deepcopy(encoded)
    bad["changed_pixels"] = 1
    with pytest.raises(ValueError, match="inconsistent"):
        deserialize_evidence(bad)


def test_compilation_snapshots_resume_without_rewriting_earlier_evidence(tmp_path, monkeypatch):
    from tools.magicbrush_grounded_v1 import runner
    from tools.prepare_magicbrush import digest, encode

    row = {
        "id": "fixture:1",
        "split": "train",
        "source_group_id": "fixture",
        "instruction": "add a cup",
        "source_image": {},
        "target_image": {},
    }
    (tmp_path / "manifest.json").write_bytes(encode({"review_ids": [row["id"]], "records": [row]}))
    monkeypatch.setattr(runner, "identity", lambda config, output: {"test_fixture_only": True})
    monkeypatch.setattr(runner, "load_image", lambda metadata: None)
    pending = runner.compile_labels(CONFIG, tmp_path)
    previous_pointer = json.loads((tmp_path / "current-compilation.json").read_text())
    previous_bytes = (tmp_path / previous_pointer["file"]).read_bytes()
    assert pending["records"][0]["status"] == "pending_inference"
    run_identity = {"test_fixture_only": True, "runtime": {"test": True}}
    (tmp_path / "inference-identity.json").write_bytes(encode(run_identity))
    source = proposal([node("ctx", "table", [600, 100, 900, 800])], {"cup": 0.01})
    target = proposal([node("ctx2", "table", [600, 100, 900, 800]), node("new", "cup", [100, 100, 300, 400])])
    ev = evidence(source, target, [[100, 100, 300, 400]])
    ev["target_anchors"]["new"] = proof(cosine=0.6, mae=0.2, old=0.1, new=0.4)
    path = tmp_path / "evidence" / f"{runner.key(row['id'])}.json"
    runner.cached_write(
        path, digest(run_identity), row["id"], source=source, target=target, evidence=serialize_evidence(ev)
    )
    completed = runner.compile_labels(CONFIG, tmp_path)
    assert completed["candidate_pairs"] == 1
    assert completed["records"][0]["eligible_for_training"] is False
    assert completed["ready_for_main_training"] is False
    assert json.loads((tmp_path / "current-compilation.json").read_text())["file"] != previous_pointer["file"]
    assert (tmp_path / previous_pointer["file"]).read_bytes() == previous_bytes
    cached = json.loads(path.read_text())
    cached["identity_sha256"] = "tampered"
    path.write_bytes(encode(cached))
    with pytest.raises(ValueError, match="provenance differs"):
        runner.compile_labels(CONFIG, tmp_path)
