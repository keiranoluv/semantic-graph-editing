"""B2: offline, hash-pinned ConceptNet 5.7 VG150 lexical-candidate graph.

Usage: python -m scripts.build_conceptnet_graph --help
No network access and no WordNet/ConceptNet sense identity assumptions.
"""

import argparse
import json
import os
import tempfile
from pathlib import Path

from src.knowledge.conceptnet_builder import extract_assertions


def write_outputs(output_dir: Path, outputs: dict[str, bytes], *, dry_run: bool = False) -> None:
    """Preflight ALL paths; atomic create without replacing differing files."""
    directory = output_dir.resolve(strict=False)
    for parent in (output_dir, *output_dir.parents):
        if parent.is_symlink():
            raise FileExistsError(f"Refusing symlink directory: {parent}")
    for name, data in outputs.items():
        target = output_dir / name
        if target.is_symlink() or (target.exists() and target.read_bytes() != data):
            raise FileExistsError(f"Refusing to replace different existing output: {target}")
    if dry_run:
        print(f"[DRY RUN] Would verify/write {len(outputs)} artifacts under {directory}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in outputs.items():
        target = output_dir / name
        if target.exists():
            print(f"[UNCHANGED] {target}")
            continue
        fd, tmp = tempfile.mkstemp(dir=output_dir, prefix=f".{name}.")
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
            os.link(tmp, target)  # Fails rather than clobber on unexpected race.
        finally:
            os.unlink(tmp)
        print(f"[WRITTEN] {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assertions", type=Path, required=True, help="Local ConceptNet 5.7 TSV or .gz snapshot")
    parser.add_argument("--expected-sha256", required=True, help="Explicitly pinned SHA256 of exact input bytes")
    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path("data/knowledge/wordnet-ai-draft-v1/mapping/wordnet_mapping.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/knowledge/conceptnet-b2-v1"))
    parser.add_argument("--min-weight", type=float, default=1.0)
    parser.add_argument("--max-edges", type=int, default=50000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    outputs, metadata = extract_assertions(
        mapping,
        args.assertions,
        expected_sha256=args.expected_sha256,
        min_weight=args.min_weight,
        max_edges=args.max_edges,
    )
    write_outputs(args.output_dir, outputs, dry_run=args.dry_run)
    print(json.dumps(metadata, indent=2))
    print("[B2 DRAFT] Lexical links are candidates, NOT WordNet synset alignments or human-verified training data.")


if __name__ == "__main__":
    main()
