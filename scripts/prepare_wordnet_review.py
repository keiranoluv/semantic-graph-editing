"""Rank WordNet sense candidates or compile explicitly reviewed overrides.

Examples:
    python -m scripts.prepare_wordnet_review rank --dry-run
    python -m scripts.prepare_wordnet_review rank
    python -m scripts.prepare_wordnet_review compile \\
        --review-csv data/knowledge/wordnet-review-v1/wordnet_review.csv
"""

import argparse
import json
from pathlib import Path

from scripts.prepare_knowledge import _save_outputs
from src.knowledge.sense_review import compile_reviews, rank_mapping, review_csv


def _json_bytes(data: dict) -> bytes:
    return (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    rank = sub.add_parser("rank", help="Create a read-only ranking and editable review CSV")
    rank.add_argument("--mapping", type=Path, default=Path("data/knowledge/wordnet-vg150/wordnet_mapping.json"))
    rank.add_argument("--output-dir", type=Path, default=Path("data/knowledge/wordnet-review-v1"))
    rank.add_argument("--dry-run", action="store_true")

    compile_parser = sub.add_parser("compile", help="Compile only explicit human accept decisions")
    compile_parser.add_argument(
        "--mapping", type=Path, default=Path("data/knowledge/wordnet-vg150/wordnet_mapping.json")
    )
    compile_parser.add_argument("--review-csv", type=Path, required=True)
    compile_parser.add_argument("--base-overrides", type=Path, default=Path("configs/knowledge/wordnet_overrides.json"))
    compile_parser.add_argument(
        "--output", type=Path, default=Path("configs/knowledge/wordnet_overrides.reviewed.json")
    )
    compile_parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    if args.action == "rank":
        try:
            from nltk.corpus import wordnet

            wordnet.ensure_loaded()
        except (ImportError, LookupError):
            parser.error("Install nltk and WordNet corpus before ranking")
        ranking = rank_mapping(mapping, wordnet=wordnet)
        _save_outputs(
            args.output_dir,
            {"wordnet_rankings.json": _json_bytes(ranking), "wordnet_review.csv": review_csv(ranking)},
            dry_run=args.dry_run,
        )
        print(f"[A2] {len(ranking['rows'])} classes ranked (NOT automatically approved)")
    else:
        base = json.loads(args.base_overrides.read_text(encoding="utf-8"))
        new_overrides, report = compile_reviews(mapping, args.review_csv.read_bytes(), base)
        _save_outputs(args.output.parent, {args.output.name: _json_bytes(new_overrides)}, dry_run=args.dry_run)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
