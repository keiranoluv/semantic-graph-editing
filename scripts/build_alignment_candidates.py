"""B.2.2: cross-check WordNet/ConceptNet VG150 candidate links (NOT SameAs).

Run: python -m scripts.build_alignment_candidates --help
Reads only local draft knowledge artifacts. No network, training, GT predicates,
or human review claims.
"""

import argparse
import json
from pathlib import Path

from scripts.build_conceptnet_graph import write_outputs
from src.knowledge.alignment_candidates import load_and_build


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wordnet-mapping",
        type=Path,
        default=Path("data/knowledge/wordnet-ai-draft-v1/mapping/wordnet_mapping.json"),
    )
    parser.add_argument("--wordnet-kg", type=Path, default=Path("data/knowledge/wordnet-ai-draft-v1/kg"))
    parser.add_argument("--conceptnet-kg", type=Path, default=Path("data/knowledge/conceptnet-b2-v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/knowledge/wordnet-conceptnet-b22-draft-v1"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    outputs, metadata = load_and_build(
        wordnet_mapping=args.wordnet_mapping,
        wordnet_kg=args.wordnet_kg,
        conceptnet_kg=args.conceptnet_kg,
    )
    write_outputs(args.output_dir, outputs, dry_run=args.dry_run)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print("[B2.2 CANDIDATES ONLY] Review all pairs before any semantic identity or training integration.")


if __name__ == "__main__":
    main()
