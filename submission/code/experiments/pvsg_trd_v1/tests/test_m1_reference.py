"""Independent Python-set oracle for the full ranking/matching aggregation."""

from collections import Counter
from copy import deepcopy
from itertools import product

import numpy as np
import pytest

from experiments.pvsg_trd_v1.contracts import PROTOCOL_ID
from experiments.pvsg_trd_v1.evaluation import evaluate


def _frames(intervals):
    return {frame for start, end in intervals for frame in range(start, end + 1)}


def _key(row):
    return row["subject_id"], row["predicate_id"], row["object_id"]


def _oracle(videos, targets, predictions):
    support = Counter()
    hits = {(k, tau): Counter() for k, tau in product((20, 50, 100), (0.3, 0.5, 0.7))}
    for video, target, prediction in zip(videos, targets, predictions, strict=False):
        visible = {track["track_id"]: _frames(track["visible_intervals"]) for track in video["tracks"]}
        gt, predicted, scores = {}, {}, {}
        for row in target["relations"]:
            gt.setdefault(_key(row), set()).update(_frames(row["intervals"]))
        for row in prediction["predictions"]:
            key = _key(row)
            predicted.setdefault(key, set()).update(_frames(row["intervals"]))
            scores[key] = max(scores.get(key, 0), row["score"])
        for key in gt:
            gt[key] &= visible[key[0]] & visible[key[2]]
            assert gt[key]
            support[key[1]] += 1
        predicted = {key: frames & visible[key[0]] & visible[key[2]] for key, frames in predicted.items()}
        ranking = sorted((key for key, frames in predicted.items() if frames), key=lambda key: (-scores[key], *key))
        for (k, tau), counts in hits.items():
            for key in ranking[:k]:
                if key in gt and len(predicted[key] & gt[key]) / len(predicted[key] | gt[key]) >= tau:
                    counts[key[1]] += 1
    total = sum(support.values())
    return {
        "support": [support[p] for p in range(57)],
        "metrics": {
            (k, tau): (
                sum(counts.values()) / total * 100,
                np.mean([counts[p] / count * 100 for p, count in support.items()]),
                [counts[p] for p in range(57)],
            )
            for (k, tau), counts in hits.items()
        },
    }


@pytest.mark.parametrize("seed", range(12))
def test_evaluator_matches_set_reference_with_duplicates_and_visibility(seed):
    rng = np.random.default_rng(seed)
    videos, targets, predictions = [], [], []
    ids = (7, 12, 30, 99)
    keys = [(s, p, o) for s, p, o in product(ids, range(9), ids) if s != o]
    for video_index in range(2):
        # Singleton intervals deliberately exercise union/adjacency handling.
        tracks = [
            {
                "track_id": track_id,
                "class_id": i,
                "is_thing": True,
                "visible_intervals": [[t, t] for t in range(17) if t == 0 or rng.random() > 0.2],
            }
            for i, track_id in enumerate(ids)
        ]
        video = {
            "schema_version": 1,
            "kind": "gt_track_video",
            "video_id": f"random_{video_index}",
            "source": "vidor" if video_index == 0 else "ego4d",
            "num_frames": 17,
            "fps": 5,
            "height": 2,
            "width": 2,
            "tracks": tracks,
            "mask_reference": "synthetic://set-oracle",
        }
        gt_rows = []
        for choice in rng.choice(len(keys), size=8 + video_index * 20, replace=False):
            s, p, o = keys[choice]
            gt_rows.append(
                {
                    "subject_id": s,
                    "predicate_id": p,
                    "object_id": o,
                    "intervals": [[0, 0], [int(rng.integers(1, 9)), int(rng.integers(9, 17))]],
                }
            )
        gt_rows += deepcopy(gt_rows[:3])
        predicted_rows = [dict(deepcopy(row), score=float(rng.choice([0.2, 0.5, 0.9]))) for row in gt_rows[:12]]
        for choice in rng.choice(len(keys), size=100, replace=False):
            s, p, o = keys[choice]
            predicted_rows.append(
                {
                    "subject_id": s,
                    "predicate_id": p,
                    "object_id": o,
                    "intervals": [[int(rng.integers(0, 9)), int(rng.integers(9, 17))]],
                    "score": float(rng.choice([0.2, 0.5, 0.9])),
                }
            )
        rng.shuffle(predicted_rows)
        videos.append(video)
        targets.append(
            {
                "schema_version": 1,
                "kind": "relation_targets",
                "video_id": video["video_id"],
                "num_frames": 17,
                "relations": gt_rows,
            }
        )
        predictions.append(
            {
                "schema_version": 1,
                "kind": "relation_predictions",
                "protocol_id": PROTOCOL_ID,
                "video_id": video["video_id"],
                "num_frames": 17,
                "predictions": predicted_rows,
            }
        )
    expected = _oracle(videos, targets, predictions)
    result = evaluate(videos, targets, predictions)
    assert result["support_per_class"] == expected["support"]
    for (k, tau), (recall, mean_recall, class_hits) in expected["metrics"].items():
        actual = result["metrics"][str(tau)][str(k)]
        assert actual["R"] == pytest.approx(recall)
        assert actual["mR"] == pytest.approx(mean_recall)
        assert actual["hits_per_class"] == class_hits
    for tau in (0.3, 0.5, 0.7):
        recalls = [result["metrics"][str(tau)][str(k)]["R"] for k in (20, 50, 100)]
        assert recalls == sorted(recalls)
    for k in (20, 50, 100):
        recalls = [result["metrics"][str(tau)][str(k)]["R"] for tau in (0.3, 0.5, 0.7)]
        assert recalls == sorted(recalls, reverse=True)
