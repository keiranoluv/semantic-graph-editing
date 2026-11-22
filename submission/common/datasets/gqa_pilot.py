"""Pinned GQA pilot cohorts and verified, frozen region-feature datasets."""

import hashlib
import json


def select_cohort(manifest, manifest_digest, train_size=512, val_size=128, seed=42):
    splits = {}
    for split, size in (("train", train_size), ("val", val_size)):
        ids = manifest["splits"][split]
        if not 0 < size <= len(ids) or len(set(ids)) != len(ids):
            raise ValueError("Invalid cohort size or duplicate image IDs")
        splits[split] = sorted(
            ids, key=lambda i: (hashlib.sha256(f"gqa-pilot-v1:{seed}:{split}:{i}".encode()).hexdigest(), i)
        )[:size]
    if set(splits["train"]) & set(splits["val"]):
        raise ValueError("Overlapping pilot splits")
    return {
        "schema_version": 1,
        "dataset": manifest["dataset"],
        "manifest_sha256": manifest_digest,
        "selection": "first N by SHA256(gqa-pilot-v1:seed:split:image_id), without filtering by annotations",
        "seed": seed,
        "splits": splits,
        "test_used": False,
    }


def validate_cohort(cohort, manifest, manifest_digest):
    expected = select_cohort(
        manifest, manifest_digest, len(cohort["splits"]["train"]), len(cohort["splits"]["val"]), cohort["seed"]
    )
    if cohort != expected:
        raise ValueError("Cohort differs from pinned deterministic selection")


def record_identity(record):
    """Bind cache rows to the complete ordered normalized record, including GT."""
    payload = {
        "image_id": record.image_id,
        "size": [record.width, record.height],
        "objects": [[o.object_id, o.label_id, o.bbox_xyxy, o.attribute_ids] for o in record.objects],
        "relations": [[r.subject_id, r.object_id, r.predicate_id] for r in record.relations],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
