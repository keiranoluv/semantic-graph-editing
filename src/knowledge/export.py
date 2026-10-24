"""C.2 draft WordNet feature export with explicit VG150 class alignment."""

import json
from pathlib import Path

import torch

from src.knowledge.entity_linking import _read_classes
from src.knowledge.sense_review import _validate_mapping
from src.knowledge.transe.core import load_kg, sha256


def export_wordnet(checkpoint_path: Path, kg_dir: Path, mapping_path: Path, vocabulary: Path, output: Path):
    """Export frozen features; unknown/background rows are zero with a false mask.

    This milestone deliberately supports draft only. Reviewed release needs a
    separately audited review contract; legacy linked_manual is not approval.
    """
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    kg = load_kg(kg_dir)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    manifest = checkpoint["manifest"]
    if kg.source != "wordnet" or manifest.get("kg_source") != "wordnet":
        raise ValueError("C.2 currently supports WordNet only; KG spaces remain separate")
    if manifest.get("milestone") != "C.1" or manifest.get("mode") != "draft":
        raise ValueError("C.2 requires a C.1 draft checkpoint; reviewed export is not implemented")
    if manifest.get("input_sha256") != kg.hashes:
        raise ValueError("Checkpoint/KG input SHA256 mismatch")
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    rows = _validate_mapping(mapping)
    classes, vocab_hash = _read_classes(vocabulary, 150)
    if len(rows) != 151 or mapping.get("vocabulary_sha256") != vocab_hash:
        raise ValueError("Mapping must match the exact VG150 vocabulary and 151 class rows")
    if kg.metadata.get("input_mapping_sha256") != sha256(mapping_path):
        raise ValueError("KG/mapping SHA256 mismatch")
    if mapping.get("source_version") != kg.metadata.get("source_version"):
        raise ValueError("Mapping/KG WordNet version mismatch")
    if any(rows[i]["label"] != name for i, name in classes):
        raise ValueError("Class labels do not match vocabulary IDs")

    dim = manifest.get("dim")
    weights = checkpoint["model"]["entity.weight"]
    relations = checkpoint["model"]["relation.weight"]
    if type(dim) is not int or dim <= 0 or weights.shape != (len(kg.entities), dim):
        raise ValueError("Entity embedding shape does not match KG/manifest")
    if relations.shape != (len(kg.relations), dim):
        raise ValueError("Relation embedding shape does not match KG/manifest")
    if not weights.is_floating_point() or not torch.isfinite(weights).all():
        raise ValueError("Invalid entity embedding values")
    if not torch.isfinite(weights.to(torch.float32)).all():
        raise ValueError("Entity embeddings overflow float32 export")

    explicit_ids = json.loads((kg_dir / "class_entity_ids.json").read_text(encoding="utf-8"))["class_entity_ids"]
    if not isinstance(explicit_ids, list) or len(explicit_ids) != 151:
        raise ValueError("Expected 151 explicit class_entity_ids")
    entity_lookup = {entity: i for i, entity in enumerate(kg.entities)}
    features = torch.zeros((151, dim), dtype=torch.float32)
    valid = torch.zeros(151, dtype=torch.bool)
    records = []
    for row, entity_id in zip(rows, explicit_ids, strict=True):
        class_id = row["class_id"]
        linked = class_id > 0 and row["status"] in {"linked_auto", "linked_manual"}
        synset = row.get("synset_id")
        if linked:
            if not isinstance(synset, str) or f"wn:{synset}" not in entity_lookup:
                raise ValueError(f"Linked synset missing from KG at class {class_id}")
            expected = entity_lookup[f"wn:{synset}"]
            if type(entity_id) is not int or entity_id != expected:
                raise ValueError(f"Class entity alignment mismatch at class {class_id}")
            features[class_id] = weights[entity_id].detach().to(torch.float32)
            valid[class_id] = True
        elif entity_id is not None or synset is not None:
            raise ValueError("Unresolved/background class must not have an entity or synset")
        records.append(
            {
                "class_id": class_id,
                "label": row["label"],
                "synset_id": synset,
                "entity_id": entity_id,
                "valid": linked,
                "mapping_status": row["status"],
                "review_status": "background" if class_id == 0 else "unverified" if linked else "unresolved",
                "source_curation_tier": row.get("curation_tier", mapping.get("curation_tier")),
            }
        )
    export_manifest = {
        "schema_version": 1,
        "milestone": "C.2",
        "mode": "draft",
        "interpretation": "draft_not_human_reviewed_not_final_results",
        "kg_source": "wordnet",
        "source_version": mapping["source_version"],
        "dim": dim,
        "class_count": 151,
        "valid_class_count": int(valid.sum()),
        "background_policy": "zero_vector_mask_false",
        "missing_policy": "zero_vector_mask_false",
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_manifest": manifest,
        "input_sha256": {
            **kg.hashes,
            "mapping": sha256(mapping_path),
            "vocabulary": vocab_hash,
            "class_entity_ids.json": sha256(kg_dir / "class_entity_ids.json"),
        },
        "rows": records,
    }
    output.mkdir(parents=True, exist_ok=False)
    tensor_path = output / "class_embeddings.pt"
    torch.save({"class_embeddings": features, "valid_mask": valid}, tensor_path)
    export_manifest["tensor_sha256"] = sha256(tensor_path)
    (output / "manifest.json").write_text(json.dumps(export_manifest, indent=2) + "\n", encoding="utf-8")
    return export_manifest


def load_class_features(folder: Path, *, allow_draft: bool = False):
    """Validate the portable C.2 bundle before training or inference."""
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("milestone") != "C.2"
        or manifest.get("kg_source") != "wordnet"
        or manifest.get("mode") != "draft"
    ):
        raise ValueError("Unsupported C.2 feature manifest; reviewed artifacts are not supported yet")
    if not allow_draft:
        raise ValueError("Draft features require explicit --allow-draft; human review is still pending")
    path = folder / "class_embeddings.pt"
    if sha256(path) != manifest.get("tensor_sha256"):
        raise ValueError("Export tensor SHA256 mismatch")
    data = torch.load(path, map_location="cpu", weights_only=True)
    features, valid = data["class_embeddings"], data["valid_mask"]
    if features.shape != (151, manifest["dim"]) or valid.shape != (151,) or valid.dtype != torch.bool:
        raise ValueError("Invalid class feature shapes or mask")
    if features.dtype != torch.float32 or not torch.isfinite(features).all():
        raise ValueError("Invalid class feature values")
    rows = manifest.get("rows", [])
    if len(rows) != 151 or any(type(r.get("class_id")) is not int or r["class_id"] != i for i, r in enumerate(rows)):
        raise ValueError("Invalid class feature row order")
    if [r.get("valid") for r in rows] != valid.tolist() or int(valid.sum()) != manifest.get("valid_class_count"):
        raise ValueError("Manifest/mask mismatch")
    if valid[0] or torch.count_nonzero(features[~valid]):
        raise ValueError("Background and unknown feature rows must be zero and invalid")
    return features, valid, manifest
