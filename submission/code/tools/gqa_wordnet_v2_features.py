"""Versioned feature export/load for partial GQA human review; v1 stays frozen."""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import torch

from src.datasets.gqa_full import atomic_write
from src.knowledge.transe.core import sha256
from tools.export_gqa_wordnet import export_features

SCOPE = "gqa_wordnet_v2_partial_human_sample_review"


def export_v2(checkpoint, mapping_dir, output):
    if output.exists():
        raise FileExistsError(f"Use a fresh v2 feature folder: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".gqa-v2-features-", dir=output.parent))
    try:
        # The immutable v1 exporter checks triple hashes, class/entity IDs and train support.
        temporary.rmdir()
        metadata = export_features(checkpoint, mapping_dir, temporary)
        mapping = json.loads((mapping_dir / "wordnet_mapping.json").read_text())
        review = mapping["gqa_review"]
        if review["accept_count"] != 23 or review["defer_count"] != 7 or review["all_classes_human_reviewed"]:
            raise ValueError("Expected the explicit 23-accept/7-defer GQA sample review")
        metadata.update(
            schema_version=2,
            scope=SCOPE,
            gqa_sample_review_completed=True,
            gqa_sample_review=review,
            gqa_context_human_review_completed=False,
            source_code_sha256={str(p): sha256(p) for p in (Path(__file__), Path("tools/export_gqa_wordnet.py"))},
        )
        for row in metadata["rows"]:
            original = mapping["rows"][row["class_id"]]
            row["gqa_human_review"] = original.get("gqa_human_review")
            if original.get("gqa_human_review", {}).get("decision") == "defer" and row["valid"]:
                raise ValueError("Explicitly deferred class must remain masked")
        atomic_write(temporary / "manifest.json", metadata)
        os.rename(temporary, output)
        return metadata
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/level2/gqa-wordnet-transe-v2-bs512/best.pt"))
    parser.add_argument("--mapping", type=Path, default=Path("data/knowledge/gqa-wordnet-v2"))
    parser.add_argument("--output", type=Path, default=Path("data/knowledge/gqa-wordnet-features-v2"))
    args = parser.parse_args(argv)
    result = export_v2(args.checkpoint, args.mapping, args.output)
    print(json.dumps({k: result[k] for k in ("scope", "valid_class_count", "checkpoint_epoch", "dim")}))


if __name__ == "__main__":
    main()
