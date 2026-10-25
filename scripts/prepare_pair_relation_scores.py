"""CPU-only export of ConceptNet TransE relation scores for VG class pairs."""

import argparse
from pathlib import Path

from src.knowledge.pair_relation import export_pair_scores


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("data/knowledge/c2-conceptnet-toan-v3"))
    parser.add_argument("--kg-dir", type=Path, default=Path("data/knowledge/conceptnet-reviewed-toan-v3"))
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/transe/conceptnet-toan-v3-seed42/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("data/knowledge/c2-conceptnet-relation-scores-toan-v1"))
    args = parser.parse_args(argv)
    manifest = export_pair_scores(args.features, args.kg_dir, args.checkpoint, args.output)
    print(f"[PAIR KG] scores={manifest['shape']} relations={manifest['relations']} output={args.output}")


if __name__ == "__main__":
    main()
