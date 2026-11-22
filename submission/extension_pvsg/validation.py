"""CPU-only checks for M0 contracts and provenance; no model or dataset loader.

This command reads local artifacts. It never downloads files or launches training.
The optional annotation audit reads vocabulary, metadata and split IDs only;
it does not use held-out relation labels to construct statistics or priors.
"""

import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical_sha256(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def validate_document(document, schema_name):
    schema = read_json(ROOT / "schemas" / schema_name)
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(document)


def validate_manifest(manifest, *, require_selected=False):
    validate_document(manifest, "dataset_manifest.schema.json")
    if manifest["state"] == "template":
        if require_selected:
            raise ValueError("The requested split is an unresolved template, not a selected dataset")
        return

    outer = manifest["outer_split"]
    train, heldout = outer["train_ids"], outer["heldout_ids"]
    if len(set(train)) != len(train) or len(set(heldout)) != len(heldout):
        raise ValueError("Duplicate video IDs in outer split")
    if set(train) & set(heldout):
        raise ValueError("Train/heldout overlap")
    if len(train) != outer["expected_counts"]["train"] or len(heldout) != outer["expected_counts"]["heldout"]:
        raise ValueError("Outer split counts differ from declared counts")
    ids = set(train) | set(heldout)
    if ids != set(manifest["video_sources"]):
        raise ValueError("Video source map does not cover exactly the outer split")
    split_payload = {"train_ids": sorted(train), "heldout_ids": sorted(heldout)}
    if canonical_sha256(split_payload) != outer["split_ids_sha256"]:
        raise ValueError("Split ID hash mismatch")
    internal = manifest["internal_validation"]
    if internal["state"] == "locked":
        fit, validation = internal["fit_ids"], internal["validation_ids"]
        if len(set(fit)) != len(fit) or len(set(validation)) != len(validation):
            raise ValueError("Duplicate internal split IDs")
        if set(fit) & set(validation) or (set(fit) | set(validation)) != set(train):
            raise ValueError("Internal split must partition official train only")
    if require_selected and (manifest["state"] != "locked" or not manifest["selected_by_user"]):
        raise ValueError("Verified candidate has not been selected by the user")


def validate_annotation_identity(path, manifest, vocabulary):
    raw = Path(path).read_bytes()
    if (
        len(raw) != manifest["annotation"]["bytes"]
        or hashlib.sha256(raw).hexdigest() != manifest["annotation"]["sha256"]
    ):
        raise ValueError("Annotation artifact size/hash mismatch")
    annotation = json.loads(raw)
    source_map = {}
    train, heldout = [], []
    for source, parts in annotation["split"].items():
        for key, destination in (("train", train), ("val", heldout)):
            for video_id in parts[key]:
                if video_id in source_map:
                    raise ValueError("Video appears in multiple source/split groups")
                source_map[video_id] = source
                destination.append(video_id)
    outer = manifest["outer_split"]
    if sorted(train) != outer["train_ids"] or sorted(heldout) != outer["heldout_ids"]:
        raise ValueError("Annotation split differs from manifest")
    if source_map != manifest["video_sources"]:
        raise ValueError("Annotation source map differs from manifest")
    video_ids = [video["video_id"] for video in annotation["data"]]
    if len(video_ids) != len(set(video_ids)) or set(video_ids) != set(source_map):
        raise ValueError("Annotation records do not cover split IDs exactly once")
    if any(video["meta"]["fps"] != 5 for video in annotation["data"]):
        raise ValueError("Annotation timeline is not uniformly 5 FPS")
    objects = annotation["objects"]["thing"] + annotation["objects"]["stuff"]
    if objects != vocabulary["object_classes"] or annotation["relations"] != vocabulary["predicates"]:
        raise ValueError("Annotation vocabulary order differs from frozen vocabulary")
    if vocabulary["annotation_sha256"] != manifest["annotation"]["sha256"]:
        raise ValueError("Vocabulary annotation provenance mismatch")
    return {"videos": len(video_ids), "train": len(train), "heldout": len(heldout)}
