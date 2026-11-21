import json
import subprocess
import sys
from copy import deepcopy

import numpy as np
import pytest
from jsonschema import ValidationError

from experiments.pvsg_trd_v1.contracts import PROTOCOL_ID, ROOT, normalize_predictions
from experiments.pvsg_trd_v1.decoding import WindowLogits, decode_windows, merge_window_logits
from experiments.pvsg_trd_v1.evaluation import evaluate
from experiments.pvsg_trd_v1.synthetic import perfect_fixture_windows, synthetic_video_targets


def predictions(video, rows):
    return {
        "schema_version": 1,
        "kind": "relation_predictions",
        "protocol_id": PROTOCOL_ID,
        "video_id": video["video_id"],
        "num_frames": video["num_frames"],
        "predictions": deepcopy(rows),
    }


def perfect(video, targets):
    return predictions(video, [dict(row, score=0.9) for row in targets["relations"]])


def metric(video, targets, predicted, **kwargs):
    return evaluate([video], [targets], [predicted], **kwargs)["metrics"]["0.5"]["20"]


def test_perfect_decoder_full_video_and_all_metrics():
    video, targets = synthetic_video_targets()
    windows = perfect_fixture_windows(video, targets)
    predicted = decode_windows(video, windows)
    assert predicted == decode_windows(video, list(reversed(windows)))
    assert [row["intervals"] for row in predicted["predictions"]] == [row["intervals"] for row in targets["relations"]]
    result = evaluate([video], [targets], [predicted], expected_video_ids=[video["video_id"]])
    for grid in result["metrics"].values():
        for row in grid.values():
            assert row["R"] == row["mR"] == 100
            assert row["hits"] == row["gt_tuples"] == 2
            assert row["recall_per_class"][2] is None
    assert result["diagnostics"]["mean_tiou_at_100"] == 100


@pytest.mark.parametrize("mutation", ["predicate", "reverse_pair", "wrong_known_track", "disjoint", "missing"])
def test_incorrect_predictions_have_zero_recall(mutation):
    video, targets = synthetic_video_targets()
    pred = perfect(video, targets)
    for row in pred["predictions"]:
        if mutation == "predicate":
            row["predicate_id"] = 56
        elif mutation == "reverse_pair":
            row["subject_id"], row["object_id"] = row["object_id"], row["subject_id"]
        elif mutation == "wrong_known_track":
            row["object_id"] = 30
        elif mutation == "disjoint":
            row["intervals"] = [[0, 1]]
    if mutation == "missing":
        pred["predictions"] = []
    assert metric(video, targets, pred)["R"] == 0


def test_overlong_prediction_penalized_not_clipped_to_gt():
    video, targets = synthetic_video_targets()
    for track in video["tracks"]:
        track["visible_intervals"] = [[0, 9]]
    targets["relations"] = [dict(targets["relations"][0], intervals=[[2, 3]])]
    pred = perfect(video, targets)
    pred["predictions"][0]["intervals"] = [[0, 9]]
    result = evaluate([video], [targets], [pred])
    assert result["metrics"]["0.5"]["20"]["R"] == 0
    assert result["diagnostics"]["mean_tiou_at_100"] == 20
    assert result["diagnostics"]["predicate_only"]["20"]["R"] == 100


def test_exact_tiou_threshold_is_a_hit():
    video, targets = synthetic_video_targets()
    targets["relations"] = [dict(targets["relations"][0], intervals=[[2, 3]])]
    pred = perfect(video, targets)
    pred["predictions"][0]["intervals"] = [[0, 3]]
    assert metric(video, targets, pred)["R"] == 100


def test_duplicate_union_and_max_score_before_global_top_k():
    video, targets = synthetic_video_targets()
    pred = perfect(video, targets)
    row = pred["predictions"][0]
    row["intervals"], row["score"] = [[2, 3]], 0.2
    pred["predictions"].append(dict(row, intervals=[[7, 8]], score=0.95))
    targets["relations"].append(deepcopy(targets["relations"][0]))
    normalized = normalize_predictions(pred, video)
    assert len(normalized["predictions"]) == 2
    assert normalized["predictions"][0]["score"] == 0.95
    assert normalized["predictions"][0]["intervals"] == [[2, 3], [7, 8]]
    result = evaluate([video], [targets], [pred], ks=(1, 2, 20))
    assert result["gt_tuples"] == result["predicted_tuples"] == 2
    assert [result["metrics"]["0.5"][str(k)]["R"] for k in (1, 2, 20)] == [50, 100, 100]
    assert result["official_metric_grid"] is False


def test_stable_tie_break_and_whole_video_k():
    video, targets = synthetic_video_targets()
    pred = perfect(video, targets)
    # Same score, lower predicate ID takes K=1; multiple labels are not collapsed.
    pred["predictions"].reverse()
    result = evaluate([video], [targets], [pred], ks=(1,))
    assert result["metrics"]["0.5"]["1"]["hits_per_class"][:2] == [1, 0]
    # A negative pair with greater score consumes the *same video-wide* budget.
    pred["predictions"].append(dict(pred["predictions"][0], subject_id=30, score=1.0))
    result = evaluate([video], [targets], [pred], ks=(1,))
    assert result["metrics"]["0.5"]["1"]["R"] == 0


def test_recall_is_micro_and_mean_recall_averages_supported_classes():
    a, ta = synthetic_video_targets()
    b = deepcopy(a)
    b.update(video_id="second", source="ego4d")
    ta["relations"] = [dict(ta["relations"][0], predicate_id=0, intervals=[[2, 3]])]
    tb = dict(deepcopy(ta), video_id=b["video_id"])
    tb["relations"] = [
        dict(ta["relations"][0], subject_id=s, object_id=o, predicate_id=1) for s, o in [(7, 12), (12, 7), (30, 7)]
    ]
    result = evaluate([a, b], [ta, tb], [predictions(a, []), perfect(b, tb)])
    assert result["metrics"]["0.5"]["20"]["R"] == 75
    assert result["metrics"]["0.5"]["20"]["mR"] == 50
    assert result["diagnostics"]["mean_tiou_at_100"] == 75  # Missed GT contributes zero.
    assert result["per_source"]["ego4d"]["support_per_class"][:2] == [0, 3]


def test_negative_videos_and_empty_split_report_null_not_nan():
    video, targets = synthetic_video_targets()
    pred = perfect(video, targets)
    targets["relations"] = []
    result = evaluate([video], [targets], [pred])
    assert result["metrics"]["0.5"]["20"]["R"] is None
    assert result["metrics"]["0.5"]["20"]["mR"] is None
    assert result["diagnostics"]["negative_video_false_positive_tuples"] == 2
    assert result["diagnostics"]["negative_videos_with_predictions"] == 1
    empty = evaluate([], [], [], expected_video_ids=[])
    assert empty["gt_tuples"] == empty["videos"] == 0
    json.dumps(empty, allow_nan=False)


@pytest.mark.parametrize(
    "mutation", ["missing_document", "duplicate_document", "expected_ids", "timeline", "unknown_track", "protocol"]
)
def test_evaluation_rejects_bad_identity_and_incomplete_cohorts(mutation):
    video, targets = synthetic_video_targets()
    pred = perfect(video, targets)
    docs, kwargs = [pred], {}
    if mutation == "missing_document":
        docs = []
    elif mutation == "duplicate_document":
        docs.append(deepcopy(pred))
    elif mutation == "expected_ids":
        kwargs["expected_video_ids"] = [video["video_id"], "missing_video"]
    elif mutation == "timeline":
        pred["num_frames"] = 11
    elif mutation == "unknown_track":
        pred["predictions"][0]["object_id"] = 999
    else:
        pred["protocol_id"] = "not-this-protocol"
    with pytest.raises((ValueError, ValidationError)):
        evaluate([video], [targets], docs, **kwargs)


def test_decoder_means_logits_before_sigmoid_and_is_order_independent():
    video, targets = synthetic_video_targets()
    base = perfect_fixture_windows(video, targets)[0]
    pairs = base.pairs
    windows = []
    for value in (6.0, -2.0, -2.0):
        z = np.full((len(pairs), 10, 57), -20.0)
        z[0, :, 0] = value
        windows.append(WindowLogits(pairs, np.arange(10), z))
    merged = merge_window_logits(video, windows)
    np.testing.assert_array_equal(merged["logits"], merge_window_logits(video, windows[::-1])["logits"])
    pred = decode_windows(video, windows)
    assert len(pred["predictions"]) == 1
    assert pred["predictions"][0]["intervals"] == [[0, 4], [7, 9]]
    assert pred["predictions"][0]["score"] == pytest.approx(1 / (1 + np.exp(-2 / 3)))


def test_decoder_exact_zero_threshold_is_active_and_multi_label():
    video, targets = synthetic_video_targets()
    pairs = perfect_fixture_windows(video, targets)[0].pairs
    pred = decode_windows(video, [WindowLogits(pairs, np.arange(10), np.zeros((6, 10, 57)))])
    assert len(pred["predictions"]) == 6 * 57
    assert all(row["score"] == 0.5 for row in pred["predictions"])


@pytest.mark.parametrize(
    "mutation",
    ["missing_tail", "missing_negative_pair", "duplicate_pair", "duplicate_frame", "frame_offset", "nan", "shape"],
)
def test_decoder_rejects_incomplete_or_misaligned_windows(mutation):
    video, targets = synthetic_video_targets()
    windows = perfect_fixture_windows(video, targets)
    first = windows[0]
    if mutation == "missing_tail":
        windows = windows[:1]
    elif mutation == "missing_negative_pair":
        windows = [WindowLogits(w.pairs[:-1], w.frame_indices, w.logits[:-1]) for w in windows]
    elif mutation == "duplicate_pair":
        first.pairs[1] = first.pairs[0]
    elif mutation == "duplicate_frame":
        first.frame_indices[1] = first.frame_indices[0]
    elif mutation == "frame_offset":
        first.frame_indices[0] = 10
    elif mutation == "nan":
        first.logits[0, 0, 0] = np.nan
    else:
        windows[0] = WindowLogits(first.pairs, first.frame_indices, first.logits[:, :, :56])
    with pytest.raises(ValueError):
        decode_windows(video, windows)


def test_no_candidate_decoder_allows_empty_predictions():
    video, targets = synthetic_video_targets()
    for track in video["tracks"]:
        track["visible_intervals"] = []
    assert decode_windows(video, [])["predictions"] == []


def test_cli_demo_and_evaluation_are_cpu_only(tmp_path):
    path = tmp_path / "demo.json"
    subprocess.run(
        [sys.executable, "-m", "experiments.pvsg_trd_v1.cli", "demo", "--output", str(path)],
        check=True,
        capture_output=True,
    )
    result = json.loads(path.read_text())
    assert result["demo"]["synthetic_only"] and not result["demo"]["model_run"]
    assert result["metrics"]["0.5"]["20"]["R"] == 100
    evaluate_cmd = [
        sys.executable,
        "-m",
        "experiments.pvsg_trd_v1.cli",
        "evaluate",
        "--videos",
        str(ROOT / "examples" / "synthetic_video.json"),
        "--targets",
        str(ROOT / "examples" / "synthetic_targets.json"),
        "--predictions",
        str(ROOT / "examples" / "synthetic_predictions.json"),
        "--output",
        str(tmp_path / "evaluation.json"),
    ]
    subprocess.run(evaluate_cmd, check=True, capture_output=True)
    evaluation = json.loads((tmp_path / "evaluation.json").read_text())
    assert evaluation["metrics"]["0.5"]["20"]["R"] == 50
    assert evaluation["dataset"]["scope"] == "custom_cohort"
    assert set(evaluation["input_sha256"]) == {"videos", "targets", "predictions"}
    bad_cohort = subprocess.run(
        evaluate_cmd
        + ["--manifest", str(ROOT / "manifests" / "pvsg_author_release_338_62.lock.json"), "--split", "test"],
        capture_output=True,
        text=True,
    )
    assert bad_cohort.returncode != 0 and "expected split IDs" in bad_cohort.stderr
    subprocess.run(
        [sys.executable, "-c", "import sys; import experiments.pvsg_trd_v1.cli; assert 'torch' not in sys.modules"],
        check=True,
    )
