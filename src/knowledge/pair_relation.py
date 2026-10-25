"""Frozen ConceptNet TransE relation scores for ordered VG object-class pairs."""

import json
from pathlib import Path

import torch

from src.knowledge.export import load_class_features
from src.knowledge.transe.core import sha256


def export_pair_scores(features_dir: Path, kg_dir: Path, checkpoint_path: Path, output_dir: Path) -> dict:
    """Export train-independent C.1 scores; VG predicate labels are never read."""
    if output_dir.exists():
        raise FileExistsError(output_dir)
    features, valid, source = load_class_features(features_dir)
    if source.get("kg_source") != "conceptnet" or source.get("mode") != "reviewed":
        raise ValueError("Expected reviewed ConceptNet C.2 features")
    if sha256(checkpoint_path) != source.get("checkpoint_sha256"):
        raise ValueError("C.1 checkpoint differs from C.2 source")
    for name in ("relations.json", "class_entity_ids.json"):
        if sha256(kg_dir / name) != source["input_sha256"][name]:
            raise ValueError(f"KG source differs from C.2 export: {name}")
    relations = json.loads((kg_dir / "relations.json").read_text(encoding="utf-8"))["relations"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    weights = checkpoint["model"]["relation.weight"].detach().float()
    if weights.shape != (len(relations), features.shape[1]) or not torch.isfinite(weights).all():
        raise ValueError("C.1 relation weights have wrong shape or nonfinite values")
    pair_mask = valid[:, None] & valid[None, :]
    scores = torch.zeros((len(valid), len(valid), len(relations)), dtype=torch.float32)
    means, scales = [], []
    for relation_id, relation in enumerate(weights):
        raw = -(features[:, None, :] + relation - features[None, :, :]).norm(dim=-1)
        mean = raw[pair_mask].mean()
        scale = raw[pair_mask].std(unbiased=False).clamp_min(1e-6)
        scores[:, :, relation_id] = torch.where(pair_mask, (raw - mean) / scale, 0)
        means.append(float(mean))
        scales.append(float(scale))
    output_dir.mkdir(parents=True, exist_ok=False)
    tensor_path = output_dir / "scores.pt"
    torch.save({"scores": scores, "valid_mask": valid, "relations": relations}, tensor_path)
    manifest = {
        "schema_version": 1,
        "source": "reviewed_conceptnet_transe_c1",
        "features_manifest_sha256": sha256(features_dir / "manifest.json"),
        "checkpoint_sha256": sha256(checkpoint_path),
        "relations_sha256": sha256(kg_dir / "relations.json"),
        "class_entity_ids_sha256": sha256(kg_dir / "class_entity_ids.json"),
        "tensor_sha256": sha256(tensor_path),
        "shape": list(scores.shape),
        "relations": relations,
        "raw_score": "negative_l2(entity_subject + relation - entity_object)",
        "normalization": "mean_and_population_std_over_valid_ordered_class_pairs_per_relation",
        "means": means,
        "scales": scales,
        "vg_predicate_labels_used": False,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


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
