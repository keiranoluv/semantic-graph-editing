"""Bind Toan's official 150-synset selection to exact WordNet KG inputs."""

import argparse
import json
from pathlib import Path

from src.knowledge.transe.core import _check_batch_review, load_kg, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kg-dir", type=Path, default=Path("data/knowledge/wordnet-kg-toan-v1"))
    parser.add_argument(
        "--mapping", type=Path, default=Path("data/knowledge/wordnet-reviewed-toan-v1/wordnet_mapping.json")
    )
    parser.add_argument(
        "--overrides", type=Path, default=Path("configs/knowledge/wordnet_overrides.reviewed-toan-v1.json")
    )
    parser.add_argument(
        "--selections", type=Path, default=Path("configs/knowledge/level1_verified_synsets_toan_2026-10-25.txt")
    )
    parser.add_argument(
        "--approval", type=Path, default=Path("configs/knowledge/level1_review_approval_toan_2026-10-25.json")
    )
    parser.add_argument("--output", type=Path, default=Path("data/knowledge/wordnet-kg-toan-v1/review_manifest.json"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    kg = load_kg(args.kg_dir)
    paths = {key: getattr(args, key).resolve() for key in ("mapping", "overrides", "selections", "approval")}
    mapping = json.loads(paths["mapping"].read_text(encoding="utf-8"))
    approval = json.loads(paths["approval"].read_text(encoding="utf-8"))
    selection_lines = paths["selections"].read_text(encoding="utf-8-sig").splitlines()
    rows = mapping.get("rows", [])
    if len(rows) != 151 or len(selection_lines) != 151:
        parser.error("Expected 150 class rows and selections")
    reviewer = approval.get("reviewer")
    manifest = {
        "schema_version": 1,
        "review_status": "human_approved_batch",
        "review_basis": "official_representative_batch_selection",
        "reviewer": reviewer,
        "review_date": approval.get("review_date"),
        "per_class_rationale_provided": False,
        "kg_source": kg.source,
        "kg_path": str(args.kg_dir.resolve()),
        "kg_sha256": kg.hashes,
        **{f"{key}_path": str(path) for key, path in paths.items()},
        **{f"{key}_sha256": sha256(path) for key, path in paths.items()},
        "reviews": [
            {
                "class_id": i,
                "label": rows[i]["label"],
                "selected_synset": selection_lines[i].strip(),
                "reviewer": reviewer,
                "evidence_reference": f"{paths['selections']}#L{i + 1}",
            }
            for i in range(1, 151)
        ],
    }
    _check_batch_review(manifest, kg, args.kg_dir)
    if args.dry_run:
        print(json.dumps({"status": "PASS", "review_status": manifest["review_status"], "class_count": 150}, indent=2))
        return
    if args.output.exists():
        raise FileExistsError(f"Review manifest already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[WRITTEN] {args.output}; official representative selections=150/150")


if __name__ == "__main__":
    main()
