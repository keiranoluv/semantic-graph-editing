from copy import deepcopy
from itertools import product

import numpy as np
import pytest
from jsonschema import ValidationError

from experiments.pvsg_trd_v1.contracts import (
    ROOT,
    normalize_predictions,
    normalize_targets,
    ordered_candidates,
    read_json,
    validate_video,
)
from experiments.pvsg_trd_v1.data import (
    parse_relation_rows,
    parse_targets,
    parse_video,
    rasterize_targets,
    split_ids,
    visibility_from_masks,
)
from experiments.pvsg_trd_v1.graphs import batch_graphs, build_graph
from experiments.pvsg_trd_v1.intervals import (
    intervals_to_mask,
    mask_to_intervals,
    normalize_intervals,
    temporal_iou,
    window_bounds,
)
from experiments.pvsg_trd_v1.synthetic import synthetic_masks, synthetic_record, synthetic_video_targets


def test_interval_roundtrip_exhaustive_and_inclusive_endpoints():
    for length in range(1, 9):
        for bits in product([False, True], repeat=length):
            mask = np.asarray(bits)
            np.testing.assert_array_equal(intervals_to_mask(mask_to_intervals(mask), length), mask)
    assert normalize_intervals([[7, 9], [0, 0], [1, 3], [2, 2]], 10) == [[0, 3], [7, 9]]
    assert mask_to_intervals(np.array([], dtype=bool)) == []
    assert temporal_iou([[0, 9]], [[2, 3]], 10) == 0.2


@pytest.mark.parametrize("spans", [[[-1, 2]], [[2, 1]], [[0, 10]], [[True, 2]], [[0.0, 2]], [[0, 1, 2]]])
def test_invalid_intervals_fail_without_clipping(spans):
    with pytest.raises(ValueError):
        normalize_intervals(spans, 10)


def test_tiou_against_independent_set_reference():
    rng = np.random.default_rng(17)
    for _ in range(100):
        first, second = rng.random((2, 37)) > 0.6
        a, b = set(np.flatnonzero(first)), set(np.flatnonzero(second))
        expected = len(a & b) / len(a | b) if a | b else 0
        assert temporal_iou(mask_to_intervals(first), mask_to_intervals(second), 37) == expected


@pytest.mark.parametrize("length", [1, 10, 64, 65, 100, 129])
def test_windows_cover_final_frame(length):
    bounds = window_bounds(length, 64, 32)
    covered = {frame for start, stop in bounds for frame in range(start, stop)}
    assert covered == set(range(length))
    assert bounds[-1][1] == length
    with pytest.raises(ValueError, match="uncovered"):
        window_bounds(length, 2, 3)


def test_adapter_uses_masks_and_keeps_negative_pairs():
    video, targets = synthetic_video_targets()
    assert video["tracks"][1]["visible_intervals"] == [[0, 4], [7, 9]]
    assert set(ordered_candidates(video)) == {(7, 12), (7, 30), (12, 7), (12, 30), (30, 7), (30, 12)}
    batch = rasterize_targets(video, targets)
    index = {tuple(pair): i for i, pair in enumerate(batch["pairs"])}
    assert batch["labels"].shape == (6, 10, 57)
    assert batch["labels"][index[7, 12], 3, :2].tolist() == [1, 1]
    assert batch["labels"][index[7, 12], 2, :2].tolist() == [1, 0]
    assert not batch["loss_mask"][index[7, 12], 5:7].any()
    assert not batch["labels"][index[12, 7]].any()
    assert not batch["labels"][index[30, 7]].any()
    assert batch["loss_mask"][index[30, 7]].all()
    cropped = rasterize_targets(video, targets, start=3, stop=8)
    np.testing.assert_array_equal(cropped["labels"], batch["labels"][:, 3:8])
    np.testing.assert_array_equal(cropped["loss_mask"], batch["loss_mask"][:, 3:8])


def test_relation_and_text_mutation_cannot_change_inputs_or_topology():
    record = synthetic_record()
    visibility = visibility_from_masks(record, synthetic_masks(), ignored_mask_ids=(0,))
    first = parse_video(record, source="vidor", visibility=visibility, mask_reference="synthetic://masks")
    record.update(
        relations="corrupt target deliberately", captions=["leaked answer"], qa_pairs=["answer"], summary="label"
    )
    second = parse_video(record, source="vidor", visibility=visibility, mask_reference="synthetic://masks")
    assert first == second
    assert not {"relations", "captions", "qa_pairs", "summary"} & set(first)
    assert ordered_candidates(first) == ordered_candidates(second)
    for field in ("nodes", "spatial_edges", "temporal_edges", "query_edges"):
        np.testing.assert_array_equal(getattr(build_graph(first), field), getattr(build_graph(second), field))


@pytest.mark.parametrize("mutation", ["missing", "extra", "float", "shape", "unknown_id", "void_is_track"])
def test_mask_audit_rejects_missing_or_ambiguous_frames(mutation):
    record, masks = synthetic_record(), synthetic_masks()
    ignored = (0,)
    if mutation == "missing":
        masks = masks[:-1]
    elif mutation == "extra":
        masks = np.concatenate((masks, masks[:1]))
    elif mutation == "float":
        masks = masks.astype(float)
    elif mutation == "shape":
        masks = masks[:, :, :2]
    elif mutation == "unknown_id":
        masks[0, 0, 0] = 99
    else:
        ignored = (0, 7)
    with pytest.raises(ValueError):
        visibility_from_masks(record, masks, ignored_mask_ids=ignored)


def test_visibility_is_mandatory_and_zero_is_not_implicitly_background():
    record = synthetic_record()
    with pytest.raises(ValueError, match="exactly"):
        parse_video(record, source="vidor", visibility={}, mask_reference="synthetic://none")
    with pytest.raises(ValueError, match="Unrecognized"):
        visibility_from_masks(record, synthetic_masks())
    empty = visibility_from_masks(record, np.zeros_like(synthetic_masks()), ignored_mask_ids=(0,))
    video = parse_video(record, source="vidor", visibility=empty, mask_reference="synthetic://empty")
    assert ordered_candidates(video) == []
    with pytest.raises(ValueError, match="empty support"):
        parse_targets(record, video)


def test_targets_apply_visibility_union_duplicates_and_preserve_multi_label():
    video, targets = synthetic_video_targets()
    targets["relations"][0]["intervals"] = [[3, 7]]
    targets["relations"].append(deepcopy(targets["relations"][0]))
    normalized = normalize_targets(targets, video)
    assert len(normalized["relations"]) == 2
    assert normalized["relations"][0]["intervals"] == [[3, 4], [7, 7]]
    assert targets["relations"][0]["intervals"] == [[3, 7]]  # Caller data was not mutated.


@pytest.mark.parametrize("mutation", ["self", "unknown", "bad_frame", "unknown_predicate"])
def test_raw_relation_errors_are_not_silently_repaired(mutation):
    raw = synthetic_record()
    if mutation == "self":
        raw["relations"][0][1] = 7
    elif mutation == "unknown":
        raw["relations"][0][1] = 99
    elif mutation == "bad_frame":
        raw["relations"][0][3] = [[2, 10]]
    else:
        raw["relations"][0][2] = "__not_in_frozen_vocabulary__"
    with pytest.raises(ValueError):
        parse_relation_rows(raw)


@pytest.mark.parametrize("mutation", ["duplicate_id", "class", "thing", "visibility", "leakage"])
def test_video_semantics_and_input_schema(mutation):
    video, _ = synthetic_video_targets()
    if mutation == "duplicate_id":
        video["tracks"][1]["track_id"] = 7
    elif mutation == "class":
        video["tracks"][0]["class_id"] = 126
    elif mutation == "thing":
        video["tracks"][0]["is_thing"] = False
    elif mutation == "visibility":
        video["tracks"][0]["visible_intervals"] = [[0, 10]]
    else:
        video["relations"] = []
    with pytest.raises((ValueError, ValidationError)):
        validate_video(video)


def test_nonfinite_prediction_scores_and_strict_json(tmp_path):
    video, targets = synthetic_video_targets()
    from experiments.pvsg_trd_v1.contracts import PROTOCOL_ID

    for value in (float("nan"), float("inf"), -float("inf")):
        doc = {
            "schema_version": 1,
            "kind": "relation_predictions",
            "protocol_id": PROTOCOL_ID,
            "video_id": video["video_id"],
            "num_frames": 10,
            "predictions": [dict(targets["relations"][0], score=value)],
        }
        with pytest.raises((ValueError, ValidationError)):
            normalize_predictions(doc, video)
    for text in ('{"score":NaN}', '{"a":1,"a":2}'):
        path = tmp_path / "bad.json"
        path.write_text(text)
        with pytest.raises(ValueError):
            read_json(path)


def test_split_roles_cannot_fallback_to_test_for_validation():
    manifest = read_json(ROOT / "manifests" / "pvsg_author_release_338_62.lock.json")
    assert len(split_ids(manifest, "train")) == 338
    assert len(split_ids(manifest, "test")) == 62
    for split in ("fit", "validation", "val"):
        with pytest.raises(ValueError):
            split_ids(manifest, split)


def test_graph_edges_temporal_gaps_and_ablation_query_invariance():
    video, _ = synthetic_video_targets()
    graph = build_graph(video)
    for s, o in graph.temporal_edges.T:
        first, second = graph.nodes[s], graph.nodes[o]
        assert first[1] == second[1]
        assert abs(first[0] - second[0]) == 1
    assert graph.temporal_edges.shape[1] == 2 * (9 + 9 + 6)
    for s, o in graph.spatial_edges.T:
        assert graph.nodes[s, 0] == graph.nodes[o, 0]
        assert graph.nodes[s, 1] != graph.nodes[o, 1]
    removed = build_graph(video, spatial=False, temporal=False)
    assert removed.spatial_edges.shape == removed.temporal_edges.shape == (2, 0)
    np.testing.assert_array_equal(removed.query_edges, graph.query_edges)
    np.testing.assert_array_equal(removed.self_edges, graph.self_edges)


def test_batch_offsets_never_connect_videos_or_overlapping_windows():
    video, _ = synthetic_video_targets()
    other = deepcopy(video)
    other["video_id"] = "another_video_with_same_track_ids"
    graphs = [build_graph(video, stop=6), build_graph(video, start=4), build_graph(other)]
    batch = batch_graphs(graphs)
    for field in ("spatial_edges", "temporal_edges", "self_edges", "query_edges"):
        edges = batch[field]
        np.testing.assert_array_equal(batch["node_batch"][edges[0]], batch["node_batch"][edges[1]])
    for i, graph in enumerate(graphs):
        mask = batch["query_batch"] == i
        np.testing.assert_array_equal(batch["query_edges"][:, mask] - batch["node_ptr"][i], graph.query_edges)
        np.testing.assert_array_equal(batch["query_track_pairs"][mask], graph.nodes[graph.query_edges, 1].T)
    assert set(batch["temporal_delta_frames"].tolist()) == {-1, 1}


def test_empty_graph_and_negative_video_targets():
    video, targets = synthetic_video_targets()
    for track in video["tracks"]:
        track["visible_intervals"] = []
    targets["relations"] = []
    batch = rasterize_targets(video, targets)
    assert batch["labels"].shape == (0, 10, 57)
    graph = batch_graphs([build_graph(video)])
    assert graph["nodes"].shape == (0, 3)
    assert graph["query_edges"].shape == (2, 0)
    assert batch_graphs([])["node_ptr"].tolist() == [0]
