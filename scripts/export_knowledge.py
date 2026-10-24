"""Export C.1 WordNet TransE checkpoint to draft class-aligned VG150 features."""

import argparse
from pathlib import Path

from src.knowledge.export import export_wordnet


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--kg-dir", type=Path, required=True)
    p.add_argument("--mapping", type=Path, required=True)
    p.add_argument("--vocabulary", type=Path, default=Path("data/raw/vg150/VG-SGG-dicts-with-attri.json"))
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    manifest = export_wordnet(args.checkpoint, args.kg_dir, args.mapping, args.vocabulary, args.output_dir)
    print(f"[DRAFT] {manifest['valid_class_count']}/150 mapped classes; dim={manifest['dim']}; {args.output_dir}")


if __name__ == "__main__":
    main()
