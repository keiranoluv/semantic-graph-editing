"""Export GQA knowledge features using the recorded partial human review."""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from common.knowledge.transe import sha256
from task2_gqa.cache import atomic_write
from task2_gqa.preprocessing.export_wordnet import export_features

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
            source_code_sha256={
                str(p): sha256(p) for p in (Path(__file__), Path("task2_gqa/preprocessing/export_wordnet.py"))
            },
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
