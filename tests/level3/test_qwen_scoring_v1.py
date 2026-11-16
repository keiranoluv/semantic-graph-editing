import copy

import pytest

from tools.level3_qwen_scoring_v1 import core
from tools.level3_qwen_scoring_v1.runner import summarize
from tools.level3_qwen_scoring_v1.scoring import align, score
from tools.level3_sgg_pretrained_v2.adapter import adapt

CFG = {"matching_iou": 0.3, "ambiguity_margin": 0.05, "min_edit_object_confidence": 0.4, "relabel_iou": 0.5}
BASE = [("person", [0, 0, 20, 50]), ("chair", [25, 30, 50, 70]), ("tree", [75, 0, 100, 90])]


def rec(objects=BASE, edges=(), size=(100, 100), confidence=0.9):
    return adapt(
        {"image_id": "test", "sha256": "s", "width": size[0], "height": size[1]},
        "react++",
        [{"label": x, "bbox_xyxy": b, "confidence": confidence} for x, b in objects],
        [{"subject": s, "object": o, "predicate": p, "confidence": 0.8} for s, p, o in edges],
        {},
        {},
    )


def run(
    source,
    target,
    kind="remove",
    name="chair",
    anchor="n001",
    region=(0.2, 0.2, 0.6, 0.8),
    status=None,
    replacement="potted plant",
):
    task = {"operation": {"kind": kind, "label": name, "replacement_label": replacement}}
    binding = {
        "source_node": anchor,
        "region": list(region),
        "binding_status": status or ("add_region" if kind == "add" else "bound_to_predicted_node"),
    }
    before = copy.deepcopy((source, target, task, binding))
    result = score(source, target, task, binding, CFG)
    assert (source, target, task, binding) == before
    for k in ["edit_score", "preservation_score", "graph_consistency"]:
        assert result[k] is None or 0 <= result[k] <= 1
    return result


def test_unchanged_image_does_not_get_half_credit_for_a_requested_edit():
    result = run(rec(), rec())
    assert result["edit_score"] == 0 and result["preservation_score"] == 1 and result["graph_consistency"] == 0


def test_remove_and_incident_relations_are_allowed_changes():
    source = rec(edges=[(0, "beside", 1), (1, "beside", 2), (0, "in front of", 2)])
    target = rec([BASE[0], BASE[2]], [(0, "in front of", 1)])
    result = run(source, target)
    assert result["graph_consistency"] == 1


def test_deleting_the_whole_scene_cannot_score_as_a_success():
    result = run(rec(), rec([]))
    assert result["edit_score"] == 1 and result["preservation_score"] == result["graph_consistency"] == 0


def test_wrong_instance_removal_not_rescued_by_class_count_drop():
    source = rec([("chair", [0, 0, 20, 40]), ("chair", [70, 0, 100, 40]), BASE[2]])
    target = rec([("chair", [0, 0, 20, 40]), BASE[2]])
    result = run(source, target, anchor="n000")
    assert result["edit_score"] == 0 and result["graph_consistency"] == 0


def test_add_existing_object_not_new():
    result = run(rec(), rec(), kind="add", anchor=None, region=(0, 0, 1, 1))
    assert result["edit_score"] == 0


def test_add_in_region_and_ignore_only_its_incident_edges():
    source = rec(edges=[(0, "beside", 2)])
    target = rec(BASE + [("dog", [30, 40, 55, 70])], [(0, "beside", 2), (3, "beside", 0)])
    assert run(source, target, kind="add", name="dog", anchor=None)["graph_consistency"] == 1


def test_add_wrong_region_is_not_fulfilled():
    target = rec(BASE + [("dog", [0, 70, 15, 95])])
    assert run(rec(), target, kind="add", name="dog", anchor=None)["edit_score"] == 0


def test_extra_unrequested_instance_is_preservation_damage():
    target = rec(BASE + [("dog", [30, 40, 55, 70]), ("dog", [0, 70, 15, 95])])
    result = run(rec(), target, kind="add", name="dog", anchor=None)
    assert (
        result["edit_score"] == 1 and result["preservation_score"] < 1 and len(result["allowed_new_target_nodes"]) == 1
    )


def test_moved_existing_object_is_not_proven_addition():
    source = rec(BASE + [("dog", [0, 70, 15, 95])])
    target = rec(BASE + [("dog", [30, 40, 55, 70])])
    result = run(source, target, kind="add", name="dog", anchor=None)
    assert result["graph_consistency"] is None and "possible_movement_or_count_instability" in result["reason_codes"]


def test_replace_at_source_region_with_expected_label():
    target = rec([BASE[0], ("potted plant", [25, 30, 50, 70]), BASE[2]])
    assert run(rec(), target, kind="replace")["graph_consistency"] == 1


def test_half_replacement_does_not_get_full_edit_credit():
    target = rec(BASE + [("potted plant", [30, 30, 45, 70])])
    result = run(rec(), target, kind="replace")
    assert result["edit_score"] == 0.5 and result["graph_consistency"] == 0.5


def test_unobserved_source_anchor_requires_abstention():
    result = run(rec([BASE[0], BASE[2]]), rec([BASE[0], BASE[2]]), anchor=None, status="source_object_not_detected")
    assert result["edit_score"] is None and result["graph_consistency"] is None


def test_imprecise_source_instruction_requires_abstention():
    assert run(rec(), rec(), status="ambiguous_instruction")["edit_score"] is None


def test_low_confidence_source_does_not_prove_removal():
    assert run(rec(confidence=0.2), rec([BASE[0], BASE[2]]))["edit_score"] is None


def test_low_confidence_new_detection_is_uncertain():
    target = rec(BASE + [("dog", [30, 40, 55, 70])])
    target["grounding"]["nodes"]["n003"]["confidence"] = 0.2
    assert run(rec(), target, kind="add", name="dog", anchor=None)["edit_score"] is None


def test_relabel_cannot_silently_prove_removal():
    target = rec([BASE[0], ("couch", [25, 30, 50, 70]), BASE[2]])
    result = run(rec(), target)
    assert result["graph_consistency"] is None and "possible_detector_relabel" in result["reason_codes"]


def test_edge_direction_matters_under_identical_object_inventory():
    source = rec(edges=[(0, "beside", 2)])
    target = rec(BASE + [("dog", [30, 40, 55, 70])], [(2, "beside", 0)])
    result = run(source, target, kind="add", name="dog", anchor=None)
    assert result["node_preservation"]["dice"] == 1 and result["edge_preservation"]["dice"] == 0
    assert result["graph_consistency"] == 0.5


def test_empty_relations_are_unobserved_not_fake_perfect_relations():
    result = run(rec(), rec())
    assert result["edge_preservation"]["dice"] is None and result["preservation_score"] == 1


def test_empty_graphs_cannot_get_perfect_preservation():
    result = run(rec([]), rec([]), kind="add", name="dog", anchor=None)
    assert result["preservation_score"] is None and result["graph_consistency"] is None


def test_local_ID_permutation_does_not_change_correspondence():
    source = rec()
    target = rec([BASE[2], BASE[0], BASE[1]])
    mapping, _, _, _ = align(source, target, CFG)
    assert mapping == {"n000": "n001", "n001": "n002", "n002": "n000"}
    assert run(source, target)["preservation_score"] == 1


def test_normalized_boxes_handle_different_image_resolution():
    scaled = [(name, [2 * v for v in box]) for name, box in BASE]
    assert run(rec(), rec(scaled, size=(200, 200)))["preservation_score"] == 1


def test_ambiguous_identical_boxes_are_flagged_and_not_scored():
    ambiguous = rec([("chair", [0, 0, 30, 40]), ("chair", [0, 0, 30, 40]), BASE[2]])
    result = run(ambiguous, ambiguous, anchor="n000")
    assert result["graph_consistency"] is None and "ambiguous_edit_anchor_alignment" in result["reason_codes"]


def test_output_cannot_overwrite_or_enter_old_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "RUNS", tmp_path / "new")
    core.new_output(tmp_path / "new/v1")
    with pytest.raises(FileExistsError):
        core.new_output(tmp_path / "new/v1")
    with pytest.raises(ValueError):
        core.new_output(tmp_path / "old")


@pytest.mark.parametrize("bad", ["stale", "human", "duplicate", "fake_pass"])
def test_assistant_review_never_masquerades_as_human_GT(bad):
    row = {
        "task_id": "x",
        "edit_assessment": "uncertain",
        "preservation_assessment": "uncertain",
        "evidence_basis": "source_and_target_images_plus_frozen_graphs",
        "observation_vi": "Ảnh còn mơ hồ",
        "graph_comment_vi": "Graph thiếu đối tượng",
    }
    pack = {"author": "assistant", "human_ground_truth": False, "score_run_receipt_sha256": "hash", "reviews": [row]}
    if bad == "stale":
        pack["score_run_receipt_sha256"] = "other"
    if bad == "human":
        pack["human_ground_truth"] = True
    if bad == "duplicate":
        pack["reviews"].append(row.copy())
    if bad == "fake_pass":
        row["edit_assessment"] = "PASS"
    with pytest.raises(ValueError):
        core.validate_assistant_review(pack, [{"task_id": "x"}], "hash")


def test_summary_reports_coverage_and_never_fabricates_GT_metrics():
    a = run(rec(), rec())
    b = run(rec(), rec(), status="ambiguous_instruction")
    report = summarize([{"scores": a}, {"scores": b}])
    assert report["pairs"] == 2 and report["graph_score_coverage"] == 0.5
    assert (
        report["human_labels_used"] == 0
        and report["human_spearman"] is None
        and report["object_or_relation_GT_F1"] is None
    )
