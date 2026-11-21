"""Semantic validation on top of the immutable M0 JSON schemas."""

import json
import math
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

from jsonschema import Draft202012Validator

from .intervals import integer, intervals_to_mask, mask_to_intervals, normalize_intervals

ROOT = Path(__file__).resolve().parent
PROTOCOL_ID = "pvsg-gt-trd-tiouv1"
NUM_PREDICATES = 57
NUM_CLASSES = 126
NUM_THINGS = 115


def read_json(path):
    def reject_constant(value):
        raise ValueError(f"Non-finite JSON value: {value}")

    def unique_keys(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(
        Path(path).read_text(encoding="utf-8"), parse_constant=reject_constant, object_pairs_hook=unique_keys
    )


@lru_cache(maxsize=None)
def _validator(name):
    return Draft202012Validator(read_json(ROOT / "schemas" / name))


def validate_video(video):
    """Return a canonical copy containing input fields only; reject extra fields."""
    _validator("video.schema.json").validate(video)
    video = deepcopy(video)
    for name in ("num_frames", "width", "height"):
        integer(video[name], name, 1)
    seen = set()
    for track in video["tracks"]:
        track_id = integer(track["track_id"], "track_id")
        class_id = integer(track["class_id"], "class_id")
        if track_id in seen:
            raise ValueError(f"Duplicate track ID: {track_id}")
        seen.add(track_id)
        if track["is_thing"] != (class_id < NUM_THINGS):
            raise ValueError("is_thing disagrees with frozen thing/stuff vocabulary")
        track["visible_intervals"] = normalize_intervals(track["visible_intervals"], video["num_frames"])
    video["tracks"].sort(key=lambda track: track["track_id"])
    return video


def visibility_masks(video):
    return {
        track["track_id"]: intervals_to_mask(track["visible_intervals"], video["num_frames"])
        for track in video["tracks"]
    }


def ordered_candidates(video):
    """Candidates depend only on GT objects/visibility, including negative pairs."""
    video = validate_video(video)
    masks = visibility_masks(video)
    return [(s, o) for s in sorted(masks) for o in sorted(masks) if s != o and (masks[s] & masks[o]).any()]


def relation_key(relation):
    return relation["subject_id"], relation["predicate_id"], relation["object_id"]


def _normalize_relations(document, video, prediction):
    video = validate_video(video)
    _validator("predictions.schema.json" if prediction else "targets.schema.json").validate(document)
    integer(document["num_frames"], "num_frames", 1)
    if (document["video_id"], document["num_frames"]) != (video["video_id"], video["num_frames"]):
        raise ValueError("Relation document video identity/timeline mismatch")
    visibility = visibility_masks(video)
    field = "predictions" if prediction else "relations"
    grouped = {}
    for row in document[field]:
        for name in ("subject_id", "object_id", "predicate_id"):
            integer(row[name], name)
        s, _, o = relation_key(row)
        if s == o or s not in visibility or o not in visibility:
            raise ValueError("Relation refers to a self pair or unknown track")
        if prediction and not math.isfinite(row["score"]):
            raise ValueError("Prediction score must be finite")
        spans = normalize_intervals(row["intervals"], video["num_frames"])
        key = relation_key(row)
        if key not in grouped:
            grouped[key] = deepcopy(row)
            grouped[key]["intervals"] = spans
        else:
            grouped[key]["intervals"].extend(spans)
            if prediction:
                grouped[key]["score"] = max(grouped[key]["score"], row["score"])
    rows = []
    for (s, _, o), row in sorted(grouped.items()):
        support = intervals_to_mask(row["intervals"], video["num_frames"]) & visibility[s] & visibility[o]
        if not support.any():
            if prediction:
                continue
            raise ValueError(f"GT tuple {relation_key(row)} has empty support after visibility filtering")
        row["intervals"] = mask_to_intervals(support)
        rows.append(row)
    if prediction:
        rows.sort(key=lambda row: (-row["score"], *relation_key(row)))
    result = deepcopy(document)
    result[field] = rows
    return result


def normalize_targets(targets, video):
    return _normalize_relations(targets, video, prediction=False)


def normalize_predictions(predictions, video):
    return _normalize_relations(predictions, video, prediction=True)
