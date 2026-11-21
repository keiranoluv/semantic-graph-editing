"""PVSG adapter with separate input/target APIs and explicit mask visibility."""

import numpy as np
from jsonschema import ValidationError

from extension_pvsg.contracts import NUM_PREDICATES, ROOT, normalize_targets, ordered_candidates, read_json, validate_video, visibility_masks
from extension_pvsg.intervals import integer, intervals_to_mask, mask_to_intervals, normalize_intervals
from extension_pvsg.validate_m0 import validate_annotation_identity, validate_manifest

VOCABULARY = read_json(ROOT / "provenance" / "vocabulary.json")
CLASS_IDS = {name: index for index, name in enumerate(VOCABULARY["object_classes"])}
PREDICATE_IDS = {name: index for index, name in enumerate(VOCABULARY["predicates"])}


def split_ids(manifest, split):
    validate_manifest(manifest, require_selected=True)
    if split in ("train", "test"):
        return list(manifest["outer_split"]["train_ids" if split == "train" else "heldout_ids"])
    if split not in ("fit", "validation"):
        raise ValueError("Split must be train, test, fit or validation")
    internal = manifest["internal_validation"]
    if internal["state"] != "locked":
        raise ValueError("Internal-validation IDs are pending M2; refusing to use upstream test for tuning")
    return list(internal["fit_ids" if split == "fit" else "validation_ids"])


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


def parse_relation_rows(record):
    """Parse raw supervision structurally; visibility filtering is a separate step."""
    track_ids = [track["track_id"] for track in _object_table(record)]
    if len(track_ids) != len(set(track_ids)):
        raise ValueError("Duplicate raw object IDs")
    rows = []
    for row in record["relations"]:
        if len(row) != 4:
            raise ValueError("Raw relation must be [subject_id, object_id, predicate, intervals]")
        s, o, predicate, spans = row
        integer(s, "subject_id")
        integer(o, "object_id")
        if s == o or s not in track_ids or o not in track_ids:
            raise ValueError("Raw relation refers to a self pair or unknown track")
        if predicate not in PREDICATE_IDS:
            raise ValueError(f"Unknown predicate: {predicate}")
        spans = normalize_intervals(spans, record["meta"]["num_frames"])
        if not spans:
            raise ValueError("Raw GT relation has no intervals")
        rows.append({"subject_id": s, "object_id": o, "predicate_id": PREDICATE_IDS[predicate], "intervals": spans})
    return {
        "schema_version": 1,
        "kind": "relation_targets",
        "video_id": record["video_id"],
        "num_frames": record["meta"]["num_frames"],
        "relations": rows,
    }


def parse_targets(record, video):
    return normalize_targets(parse_relation_rows(record), video)


def visibility_from_masks(record, masks, *, ignored_mask_ids=()):
    """Stream decoded 2-D integer ID masks, one per annotation frame.

    The caller must specify any void IDs explicitly; no implicit background=0.
    RGB/packed panoptic decoding and media path alignment belong to M2.
    """
    ids = [track["track_id"] for track in _object_table(record)]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate raw object IDs")
    ignored = set(ignored_mask_ids)
    if ignored & set(ids):
        raise ValueError("An ignored mask ID is also an annotated object")
    meta = record["meta"]
    frames = integer(meta["num_frames"], "num_frames", 1)
    visible = {track_id: np.zeros(frames, dtype=bool) for track_id in ids}
    count = 0
    for t, mask in enumerate(masks):
        mask = np.asarray(mask)
        if t >= frames:
            raise ValueError("Too many mask frames")
        if mask.shape != (meta["height"], meta["width"]) or mask.dtype.kind not in "iu":
            raise ValueError("Mask must have metadata H/W and integer track IDs")
        present = set(map(int, np.unique(mask))) - ignored
        if not present <= set(ids):
            raise ValueError(f"Unrecognized mask IDs: {sorted(present - set(ids))}")
        for track_id in present:
            visible[track_id][t] = True
        count += 1
    if count != frames:
        raise ValueError(f"Expected {frames} mask frames, got {count}")
    return {track_id: mask_to_intervals(timeline) for track_id, timeline in visible.items()}


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


class AnnotationIndex:
    """Local pinned annotation access; explicit split selection, no download."""

    def __init__(self, annotation_path, manifest_path=None):
        self.manifest = read_json(manifest_path or ROOT / "manifests" / "pvsg_author_release_338_62.lock.json")
        validate_manifest(self.manifest, require_selected=True)
        validate_annotation_identity(annotation_path, self.manifest, VOCABULARY)
        annotation = read_json(annotation_path)
        self._records = {video["video_id"]: video for video in annotation["data"]}

    def ids(self, split):
        return split_ids(self.manifest, split)

    def video(self, video_id, *, visibility, mask_reference):
        return parse_video(
            self._records[video_id],
            source=self.manifest["video_sources"][video_id],
            visibility=visibility,
            mask_reference=mask_reference,
        )

    def targets(self, video):
        return parse_targets(self._records[video["video_id"]], video)

    def audit_train_structure(self):
        """Check raw train metadata/targets only; never invent visible frames."""
        errors = []
        for video_id in self.ids("train"):
            record = self._records[video_id]
            try:
                empty_visibility = {track["track_id"]: [] for track in _object_table(record)}
                self.video(video_id, visibility=empty_visibility, mask_reference="audit://visibility-not-available")
                parse_relation_rows(record)
            except (ValueError, KeyError, TypeError, ValidationError) as error:
                errors.append({"video_id": video_id, "error": str(error)})
        return {
            "scope": "train_structure_only",
            "videos": len(self.ids("train")),
            "errors": errors,
            "videos_with_errors": len(errors),
            "first_error_per_video_only": True,
            "annotation_sha256": self.manifest["annotation"]["sha256"],
            "media_visibility_checked": False,
            "heldout_targets_read_for_statistics": False,
        }
