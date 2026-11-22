"""Frozen ConceptNet TransE relation scores for ordered VG object-class pairs."""

import json
from pathlib import Path

import torch
from common.knowledge.export import load_class_features
from common.knowledge.transe.core import sha256


def load_pair_scores(folder: Path, features_dir: Path) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Validate portable scores and their C.2 alignment before training."""
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("source") != "reviewed_conceptnet_transe_c1":
        raise ValueError("Unsupported pair-score manifest")
    if sha256(features_dir / "manifest.json") != manifest.get("features_manifest_sha256"):
        raise ValueError("Pair scores use different C.2 features")
    path = folder / "scores.pt"
    if sha256(path) != manifest.get("tensor_sha256"):
        raise ValueError("Pair-score tensor hash mismatch")
    data = torch.load(path, map_location="cpu", weights_only=True)
    features, valid, _ = load_class_features(features_dir)
    scores = data["scores"]
    if (
        scores.shape != (features.shape[0], features.shape[0], len(manifest["relations"]))
        or list(scores.shape) != manifest.get("shape")
        or scores.dtype != torch.float32
        or not torch.isfinite(scores).all()
        or not torch.equal(data["valid_mask"], valid)
        or data["relations"] != manifest["relations"]
    ):
        raise ValueError("Pair-score tensor content does not match manifest or C.2")
    if bool((scores[~(valid[:, None] & valid[None, :])]).any()):
        raise ValueError("Unlinked class pairs must have zero scores")
    return scores, valid, manifest
