"""Produce deterministic WordNet links for the VG150 object vocabulary.

Run: python -m scripts.prepare_knowledge --root data/raw/vg150
Install first: pip install nltk && python -m nltk.downloader wordnet
"""

import argparse
import csv
import io
import json
import os
import tempfile
from pathlib import Path

from src.knowledge.entity_linking import link_vocabulary


def _serialize(mapping: dict, report: dict) -> dict[str, bytes]:
    """Return reproducible files; write review items independently of model input."""
    review = io.StringIO(newline="")
    writer = csv.writer(review, lineterminator="\n")
    writer.writerow(["class_id", "label", "status", "candidates", "synset_id"])
    for row in mapping["rows"][1:]:
        if row["status"] not in ("ambiguous", "unmatched"):
            continue
        writer.writerow(
            [row["class_id"], row["label"], row["status"], ";".join(c["synset_id"] for c in row["candidates"]), ""]
        )
    return {
        "wordnet_mapping.json": (json.dumps(mapping, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
        "wordnet_coverage.json": (json.dumps(report, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
        "wordnet_review.csv": review.getvalue().encode("utf-8"),
    }


def _save_outputs(out_dir: Path, outputs: dict[str, bytes], *, dry_run: bool) -> None:
    """Idempotent saving: existing mismatched artifacts are never overwritten."""
    for name, content in outputs.items():
        path = out_dir / name
        if path.is_symlink():
            raise FileExistsError(f"Refusing symlink output: {path}")
        if path.exists() and path.read_bytes() != content:
            raise FileExistsError(f"Output differs from existing file: {path}. Use a fresh output directory")
    if dry_run:
        print(f"[DRY RUN] Would verify/write {len(outputs)} outputs under {out_dir}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, content in outputs.items():
        destination = out_dir / name
        if destination.exists():
            print(f"[UNCHANGED] {destination}")
            continue
        fd, tmp_name = tempfile.mkstemp(prefix=f".{name}.", dir=out_dir)
        try:
            with os.fdopen(fd, "wb") as file:
                file.write(content)
            # Avoid overwriting a newly created file in another process.
            os.link(tmp_name, destination)
        finally:
            os.unlink(tmp_name)
        print(f"[WRITTEN] {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/vg150"))
    parser.add_argument("--vocabulary", type=Path, default=None)
    parser.add_argument("--overrides", type=Path, default=Path("configs/knowledge/wordnet_overrides.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/knowledge/wordnet-vg150"))
    parser.add_argument("--expected-classes", type=int, default=150)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        import nltk
        from nltk.corpus import wordnet as wordnet

        wordnet.ensure_loaded()  # never download implicitly
    except (ImportError, LookupError):
        parser.error("WordNet corpus not installed; run: pip install nltk && python -m nltk.downloader wordnet")

    vocabulary_path = args.vocabulary or (args.root / "VG-SGG-dicts-with-attri.json")
    mapping, report = link_vocabulary(
        vocabulary_path,
        wordnet=wordnet,
        expected_classes=args.expected_classes,
        overrides_path=args.overrides,
        nltk_version=nltk.__version__,
    )
    _save_outputs(args.output_dir, _serialize(mapping, report), dry_run=args.dry_run)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
