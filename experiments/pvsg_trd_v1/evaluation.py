"""GT-track temporal-set-IoU evaluator. Not OpenPVSG volumetric compatibility."""

import numpy as np

from .contracts import (
    NUM_PREDICATES,
    PROTOCOL_ID,
    normalize_predictions,
    normalize_targets,
    relation_key,
    validate_video,
)
from .intervals import integer, temporal_iou

EVALUATOR_ID = "pvsg-gt-trd-tiouv1-cpu-m1"
DEFAULT_KS = (20, 50, 100)
DEFAULT_THRESHOLDS = (0.3, 0.5, 0.7)


def _index(documents, kind):
    result = {}
    for document in documents:
        video_id = document["video_id"]
        if video_id in result:
            raise ValueError(f"Duplicate {kind} video document: {video_id}")
        result[video_id] = document
    return result


def _recall(hits, support):
    supported = support > 0
    recall = np.divide(hits, support, out=np.zeros(NUM_PREDICATES, dtype=float), where=supported) * 100
    total = int(support.sum())
    return {
        "R": float(hits.sum() / total * 100) if total else None,
        "mR": float(recall[supported].mean()) if total else None,
        "hits": int(hits.sum()),
        "gt_tuples": total,
        "hits_per_class": hits.tolist(),
        "recall_per_class": [float(value) if keep else None for value, keep in zip(recall, supported, strict=False)],
    }


def _aggregate(records, ks, thresholds):
    support = np.zeros(NUM_PREDICATES, dtype=np.int64)
    hits = np.zeros((len(thresholds), len(ks), NUM_PREDICATES), dtype=np.int64)
    predicate_hits = np.zeros((len(ks), NUM_PREDICATES), dtype=np.int64)
    for record in records:
        support += record["support"]
        hits += record["hits"]
        predicate_hits += record["predicate_hits"]
    gt_count = int(support.sum())
    negative = [record for record in records if record["support"].sum() == 0]
    return {
        "videos": len(records),
        "gt_tuples": gt_count,
        "predicted_tuples": sum(record["predictions"] for record in records),
        "support_per_class": support.tolist(),
        "supported_predicates": np.flatnonzero(support).tolist(),
        "metrics": {
            str(tau): {str(k): _recall(hits[j, i], support) for i, k in enumerate(ks)}
            for j, tau in enumerate(thresholds)
        },
        "diagnostics": {
            "predicate_only": {str(k): _recall(predicate_hits[i], support) for i, k in enumerate(ks)},
            "mean_tiou_at_100": sum(record["tiou_at_100_sum"] for record in records) / gt_count * 100
            if gt_count
            else None,
            "negative_videos": len(negative),
            "negative_videos_with_predictions": sum(record["predictions"] > 0 for record in negative),
            "negative_video_false_positive_tuples": sum(record["predictions"] for record in negative),
            "negative_video_false_positive_tuples_at_k": {
                str(k): sum(min(k, record["predictions"]) for record in negative) for k in ks
            },
        },
    }


def evaluate(videos, targets, predictions, *, expected_video_ids=None, ks=DEFAULT_KS, thresholds=DEFAULT_THRESHOLDS):
    """Evaluate exactly the requested cohort; require explicit empty predictions.

    Non-default metric grids are diagnostic only and flagged in the report.
    Scores/recalls and the mean-tIoU diagnostic are reported in percent.
    """
    ks = tuple(integer(k, "K", 1) for k in ks)
    thresholds = tuple(float(tau) for tau in thresholds)
    if not ks or len(set(ks)) != len(ks) or not thresholds or len(set(thresholds)) != len(thresholds):
        raise ValueError("Metric grid must be nonempty and unique")
    if any(not np.isfinite(tau) or not 0 < tau <= 1 for tau in thresholds):
        raise ValueError("tIoU thresholds must be finite in (0,1]")
    video_map = _index(videos, "input")
    target_map = _index(targets, "target")
    prediction_map = _index(predictions, "prediction")
    if set(video_map) != set(target_map) or set(video_map) != set(prediction_map):
        raise ValueError("Input/target/prediction cohort mismatch; emit an empty prediction document for missed videos")
    if expected_video_ids is not None:
        expected = list(expected_video_ids)
        if len(expected) != len(set(expected)) or set(expected) != set(video_map):
            raise ValueError("Evaluation cohort differs from expected split IDs")
    records = []
    for video_id in sorted(video_map):
        video = validate_video(video_map[video_id])
        gt = normalize_targets(target_map[video_id], video)["relations"]
        ranked = normalize_predictions(prediction_map[video_id], video)["predictions"]
        by_key = {relation_key(row): (rank, row) for rank, row in enumerate(ranked)}
        support = np.zeros(NUM_PREDICATES, dtype=np.int64)
        hits = np.zeros((len(thresholds), len(ks), NUM_PREDICATES), dtype=np.int64)
        predicate_hits = np.zeros((len(ks), NUM_PREDICATES), dtype=np.int64)
        tiou_sum = 0.0
        for row in gt:
            predicate = row["predicate_id"]
            support[predicate] += 1
            match = by_key.get(relation_key(row))
            if match is None:
                continue
            rank, prediction = match
            iou = temporal_iou(row["intervals"], prediction["intervals"], video["num_frames"])
            if rank < 100:
                tiou_sum += iou
            for i, k in enumerate(ks):
                if rank < k:
                    predicate_hits[i, predicate] += 1
                    for j, tau in enumerate(thresholds):
                        hits[j, i, predicate] += iou >= tau
        records.append(
            {
                "video_id": video_id,
                "source": video["source"],
                "support": support,
                "hits": hits,
                "predicate_hits": predicate_hits,
                "tiou_at_100_sum": tiou_sum,
                "predictions": len(ranked),
            }
        )
    result = _aggregate(records, ks, thresholds)
    result.update(
        protocol_id=PROTOCOL_ID,
        evaluator_id=EVALUATOR_ID,
        report_scale="percent",
        evaluated_video_ids=sorted(video_map),
        cohort_checked_against_expected_ids=expected_video_ids is not None,
        official_metric_grid=ks == DEFAULT_KS and thresholds == DEFAULT_THRESHOLDS,
        per_source={
            source: _aggregate([record for record in records if record["source"] == source], ks, thresholds)
            for source in sorted({record["source"] for record in records})
        },
    )
    return result
