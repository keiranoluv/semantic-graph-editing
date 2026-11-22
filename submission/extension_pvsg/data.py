"""PVSG adapter with separate input/target APIs and explicit mask visibility."""

import numpy as np
from extension_pvsg.contracts import (
    NUM_PREDICATES,
    ROOT,
    normalize_targets,
    ordered_candidates,
    read_json,
    validate_video,
    visibility_masks,
)
from extension_pvsg.intervals import integer, intervals_to_mask

VOCABULARY = read_json(ROOT / "provenance" / "vocabulary.json")

CLASS_IDS = {name: index for index, name in enumerate(VOCABULARY["object_classes"])}

PREDICATE_IDS = {name: index for index, name in enumerate(VOCABULARY["predicates"])}


def _object_table(record):
    tracks = []
    for obj in record["objects"]:
        if obj["category"] not in CLASS_IDS:
            raise ValueError(f"Unknown object category: {obj['category']}")
        tracks.append(
            {
                "track_id": integer(obj["object_id"], "object_id"),
                "class_id": CLASS_IDS[obj["category"]],
                "is_thing": obj["is_thing"],
                "visible_intervals": [],
            }
        )
    return tracks


def parse_video(record, *, source, visibility, mask_reference):
    """Project raw annotation onto object inputs. Never read relations/text fields.

    visibility maps every integer track ID to intervals obtained from masks.
    No fallback to annotated relation spans or assumed full-video visibility.
    """
    tracks = _object_table(record)
    if set(visibility) != {track["track_id"] for track in tracks}:
        raise ValueError("Visibility must cover exactly the annotated track IDs")
    for track in tracks:
        track["visible_intervals"] = visibility[track["track_id"]]
    meta = record["meta"]
    return validate_video(
        {
            "schema_version": 1,
            "kind": "gt_track_video",
            "video_id": record["video_id"],
            "source": source,
            "num_frames": meta["num_frames"],
            "fps": meta["fps"],
            "width": meta["width"],
            "height": meta["height"],
            "tracks": tracks,
            "mask_reference": mask_reference,
        }
    )


def rasterize_targets(video, targets, *, start=0, stop=None):
    """Multi-label [P,L,57] and independent [P,L] visibility/loss mask.

    Pair order is global to the video, even when a pair is absent in this window.
    Negative pairs remain in the target tensor. No softmax background channel.
    """
    video = validate_video(video)
    targets = normalize_targets(targets, video)
    total = video["num_frames"]
    stop = total if stop is None else integer(stop, "stop", 1)
    start = integer(start, "start")
    if not start < stop <= total:
        raise ValueError("Invalid half-open window bounds")
    pairs = ordered_candidates(video)
    pair_index = {pair: i for i, pair in enumerate(pairs)}
    visibility = visibility_masks(video)
    loss_mask = np.array([(visibility[s] & visibility[o])[start:stop] for s, o in pairs], dtype=bool).reshape(
        len(pairs), stop - start
    )
    labels = np.zeros((len(pairs), stop - start, NUM_PREDICATES), dtype=np.float32)
    for row in targets["relations"]:
        labels[pair_index[row["subject_id"], row["object_id"]], :, row["predicate_id"]] = intervals_to_mask(
            row["intervals"], total
        )[start:stop]
    return {
        "pairs": np.asarray(pairs, dtype=np.int64).reshape(-1, 2),
        "frame_indices": np.arange(start, stop),
        "labels": labels,
        "loss_mask": loss_mask,
    }
