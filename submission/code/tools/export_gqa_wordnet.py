"""Export vocabulary-bound GQA TransE features without VG150 ID assumptions."""

import argparse
import json
from pathlib import Path

import torch

from scripts.prepare_knowledge import _save_outputs
from src.knowledge.entity_linking import _read_classes
from src.knowledge.sense_review import _validate_mapping
from src.knowledge.transe.core import load_kg, sha256, split_triples_transductive
from tools.prepare_gqa_wordnet import json_bytes


def export_features(checkpoint_path, mapping_dir, output):
    if output.exists():
        raise FileExistsError(f"Use a fresh feature bundle: {output}")
    kg_dir = mapping_dir / "kg"
    kg = load_kg(kg_dir)
    mapping_path = mapping_dir / "wordnet_mapping.json"
    mapping = json.loads(mapping_path.read_text())
    rows = _validate_mapping(mapping)
    classes, vocabulary_hash = _read_classes(mapping_dir / "vocabulary.json", len(rows) - 1)
    if mapping["vocabulary_sha256"] != vocabulary_hash or any(rows[i]["label"] != name for i, name in classes):
        raise ValueError("GQA mapping vocabulary/ID mismatch")
    if kg.metadata["input_mapping_sha256"] != sha256(mapping_path):
        raise ValueError("KG mapping hash mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    source = checkpoint["manifest"]
    if (
        kg.source != "wordnet"
        or source.get("kg_source") != "wordnet"
        or source.get("input_sha256") != kg.hashes
        or source.get("mode") != "draft"
        or source.get("milestone") != "C.1"
        or source.get("split_policy") != "seeded_transductive_coverage_holdout"
    ):
        raise ValueError("Expected exact draft WordNet TransE checkpoint with transductive split")
    weights = checkpoint["model"]["entity.weight"]
    if (
        weights.shape != (len(kg.entities), source["dim"])
        or weights.dtype != torch.float32
        or not torch.isfinite(weights).all()
    ):
        raise ValueError("Invalid KG entity weights")
    train, _, _ = split_triples_transductive(
        kg.triples, source["seed"], val_ratio=source["val_ratio"], test_ratio=source["test_ratio"]
    )
    trained = set(train[:, 0]) | set(train[:, 2])
    explicit = json.loads((kg_dir / "class_entity_ids.json").read_text())["class_entity_ids"]
    if len(explicit) != len(rows) or explicit[0] is not None:
        raise ValueError("Class/entity alignment length or background mismatch")
    features = torch.zeros(len(rows), source["dim"], dtype=torch.float32)
    valid = torch.zeros(len(rows), dtype=torch.bool)
    records = []
    for row, entity_id in zip(rows, explicit, strict=True):
        i, synset = row["class_id"], row["synset_id"]
        if synset is not None:
            if (
                type(entity_id) is not int
                or not 0 <= entity_id < len(kg.entities)
                or kg.entities[entity_id] != f"wn:{synset}"
            ):
                raise ValueError(f"Class/entity alignment mismatch at {i}")
            if entity_id in trained:
                features[i] = weights[entity_id]
                valid[i] = True
        elif entity_id is not None:
            raise ValueError("Unlinked class must have no KG entity")
        records.append(
            {
                "class_id": i,
                "label": row["label"],
                "synset_id": synset,
                "entity_id": entity_id,
                "valid": bool(valid[i]),
                "reason": "trained_external_entity"
                if valid[i]
                else "background"
                if i == 0
                else "isolated_in_kg_training"
                if synset is not None
                else "unresolved",
            }
        )
    if not valid.any():
        raise ValueError("No trained linked classes")
    metadata = {
        "schema_version": 1,
        "dataset": "GQA_custom_full_train_vocab",
        "scope": "draft_pilot_not_human_reviewed_GQA",
        "kg_source": "wordnet",
        "dim": source["dim"],
        "class_count": len(rows),
        "valid_class_count": int(valid.sum()),
        "gqa_manifest_sha256": json.loads((mapping_dir / "coverage.json").read_text())["sources"][
            "gqa_manifest_sha256"
        ],
        "vocabulary_sha256": vocabulary_hash,
        "mapping_sha256": sha256(mapping_path),
        "kg_sha256": kg.hashes,
        "alignment_sha256": sha256(kg_dir / "class_entity_ids.json"),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_epoch": checkpoint["epoch"],
        "embedding_training": source,
        "rows": records,
        "background_and_missing_policy": "zero_vector_and_false_valid_bit",
        "isolated_entity_policy": "zero_vector_and_false_valid_bit_no_random_untrained_vectors",
        "gqa_context_human_review_completed": False,
        "test_annotation_opened": False,
        "source_code_sha256": {str(Path(__file__).relative_to(Path.cwd())): sha256(Path(__file__))},
    }
    output.mkdir(parents=True, exist_ok=False)
    tensor_path = output / "class_embeddings.pt"
    torch.save({"class_embeddings": features, "valid_mask": valid}, tensor_path)
    metadata["tensor_sha256"] = sha256(tensor_path)
    _save_outputs(output, {"manifest.json": json_bytes(metadata)}, dry_run=False)
    return metadata


def load_features(folder, manifest_path):
    """Fail closed on changed tensors, class ordering or train vocabulary."""
    metadata = json.loads((folder / "manifest.json").read_text())
    manifest = json.loads(manifest_path.read_text())
    labels = manifest["vocabulary"]["label_to_idx"]
    if (
        metadata.get("schema_version") != 1
        or metadata.get("kg_source") != "wordnet"
        or metadata.get("gqa_manifest_sha256") != sha256(manifest_path)
        or metadata.get("class_count") != len(labels) + 1
        or metadata.get("scope") != "draft_pilot_not_human_reviewed_GQA"
    ):
        raise ValueError("GQA feature bundle scope or vocabulary mismatch")
    tensor_path = folder / "class_embeddings.pt"
    if sha256(tensor_path) != metadata["tensor_sha256"]:
        raise ValueError("Feature tensor hash mismatch")
    data = torch.load(tensor_path, map_location="cpu", weights_only=True)
    features, valid = data["class_embeddings"], data["valid_mask"]
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
        raise ValueError("Invalid feature shape, values or background/missing mask")
    rows = metadata["rows"]
    if (
        len(rows) != n
        or any(type(r["class_id"]) is not int or r["class_id"] != i for i, r in enumerate(rows))
        or any(rows[i]["label"] != label for label, i in labels.items())
        or [r["valid"] for r in rows] != valid.tolist()
    ):
        raise ValueError("GQA feature row order or label mismatch")
    return features, valid, metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/level2/gqa-wordnet-transe-draft-v1/best.pt"))
    parser.add_argument("--mapping-dir", type=Path, default=Path("data/knowledge/gqa-wordnet-v1"))
    parser.add_argument("--output", type=Path, default=Path("data/knowledge/gqa-wordnet-features-draft-v1"))
    args = parser.parse_args(argv)
    metadata = export_features(args.checkpoint, args.mapping_dir, args.output)
    print(
        json.dumps({k: metadata[k] for k in ("class_count", "valid_class_count", "dim", "checkpoint_epoch", "scope")})
    )


if __name__ == "__main__":
    main()
