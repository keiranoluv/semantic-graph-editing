"""Load and validate the frozen GQA knowledge features used by checkpoints."""

import json

import torch
from common.knowledge.transe.core import sha256

SCOPE = "gqa_wordnet_v2_partial_human_sample_review"


def load_v2(folder, manifest_path):
    metadata = json.loads((folder / "manifest.json").read_text())
    manifest = json.loads(manifest_path.read_text())
    labels = manifest["vocabulary"]["label_to_idx"]
    if (
        metadata.get("schema_version") != 2
        or metadata.get("scope") != SCOPE
        or metadata.get("gqa_manifest_sha256") != sha256(manifest_path)
        or metadata.get("class_count") != len(labels) + 1
        or metadata.get("gqa_sample_review_completed") is not True
        or metadata.get("gqa_context_human_review_completed") is not False
    ):
        raise ValueError("Wrong v2 feature scope/vocabulary/review provenance")
    path = folder / "class_embeddings.pt"
    if sha256(path) != metadata["tensor_sha256"]:
        raise ValueError("KG feature tensor hash mismatch")
    state = torch.load(path, map_location="cpu", weights_only=True)
    features, valid = state["class_embeddings"], state["valid_mask"]
    n = len(labels) + 1
    if (
        features.shape != (n, metadata["dim"])
        or features.dtype != torch.float32
        or valid.shape != (n,)
        or valid.dtype != torch.bool
        or not torch.isfinite(features).all()
        or valid[0]
        or torch.count_nonzero(features[~valid])
        or int(valid.sum()) != metadata["valid_class_count"]
    ):
        raise ValueError("Invalid v2 KG tensor/mask")
    rows = metadata["rows"]
    if (
        len(rows) != n
        or any(type(r["class_id"]) is not int or r["class_id"] != i for i, r in enumerate(rows))
        or any(rows[i]["label"] != label for label, i in labels.items())
        or [r["valid"] for r in rows] != valid.tolist()
    ):
        raise ValueError("KG feature ID/label alignment differs")
    for row in rows:
        if (row.get("gqa_human_review") or {}).get("decision") == "defer" and row["valid"]:
            raise ValueError("Deferred class unexpectedly has KG features")
    return features, valid, metadata
