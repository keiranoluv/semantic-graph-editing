"""Export reviewed ConceptNet RotatE v2 vectors as 64D VG150 class features."""

import argparse
from pathlib import Path

from src.knowledge.export import export_conceptnet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--kg-dir", type=Path, required=True)
    parser.add_argument("--review-manifest", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, default=Path("data/raw/vg150/VG-SGG-dicts-with-attri.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = export_conceptnet(
        args.checkpoint,
        args.kg_dir,
        args.vocabulary,
        args.output_dir,
        review_manifest_path=args.review_manifest,
        embedding_model="rotate",
    )
    print(f"[CONCEPTNET ROTATE] {manifest['valid_class_count']}/150 mapped classes; dim={manifest['dim']}")


if __name__ == "__main__":
    main()
