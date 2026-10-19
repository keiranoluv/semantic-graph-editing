"""Construct deterministic WordNet KG artifacts from reviewed VG150 linking.

For a first smoke test, the current partially resolved mapping may be used.
Set --require-fully-reviewed for the strict final experiment gate.
"""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

from scripts.prepare_knowledge import _save_outputs
from src.knowledge.graph_builder import build_wordnet_graph


def _json_bytes(data: object) -> bytes:
    return (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def serialize_graph(graph: dict, *, mapping_file_sha256: str) -> dict[str, bytes]:
    """Return portable entity, relation, triple and class-alignment outputs."""
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
    writer.writerow(("subject_id", "relation_id", "object_id", "provenance"))
    for item in graph["triples"]:
        writer.writerow((item["subject_id"], item["relation_id"], item["object_id"], item["provenance"]))
    metadata = {
        **graph["metadata"],
        "source": graph["source"],
        "source_version": graph["source_version"],
        "input_mapping_sha256": mapping_file_sha256,
        "schema_version": graph["schema_version"],
    }
    return {
        "entities.json": _json_bytes({"entities": graph["entities"]}),
        "relations.json": _json_bytes({"relations": graph["relations"]}),
        "class_entity_ids.json": _json_bytes({"class_entity_ids": graph["class_entity_ids"]}),
        "triples.tsv": stream.getvalue().encode("utf-8"),
        "metadata.json": _json_bytes(metadata),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, default=Path("data/knowledge/wordnet-vg150/wordnet_mapping.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/knowledge/wordnet-kg-draft-v1"))
    parser.add_argument("--max-hops", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--max-entities", type=int, default=10000)
    parser.add_argument("--max-triples", type=int, default=50000)
    parser.add_argument("--require-fully-reviewed", action="store_true")
    parser.add_argument(
        "--reviewed-overrides",
        type=Path,
        default=None,
        help="Reviewed override JSON; must SHA256-match mapping overrides",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        from nltk.corpus import wordnet

        wordnet.ensure_loaded()
    except (ImportError, LookupError):
        parser.error("Install nltk and WordNet corpus before running B1")
    raw = args.mapping.read_bytes()
    mapping = json.loads(raw)
    reviewed_ids = frozenset()
    if args.reviewed_overrides is not None:
        override_raw = args.reviewed_overrides.read_bytes()
        if hashlib.sha256(override_raw).hexdigest() != mapping.get("overrides_sha256"):
            parser.error("Reviewed override SHA256 does not match mapping provenance")
        overrides = json.loads(override_raw)
        labels = overrides.get("mappings", {})
        audits = overrides.get("review_provenance", {})
        if not isinstance(labels, dict) or not isinstance(audits, dict):
            parser.error("Reviewed override file lacks mapping and provenance dictionaries")
        ids = []
        for row in mapping["rows"][1:]:
            info = audits.get(row["label"], {})
            if (
                row["status"] == "linked_manual"
                and labels.get(row["label"]) == row.get("synset_id")
                and isinstance(info, dict)
                and info.get("reviewer")
                and info.get("class_id") == row["class_id"]
            ):
                ids.append(row["class_id"])
        reviewed_ids = frozenset(ids)
    graph = build_wordnet_graph(
        mapping,
        wordnet=wordnet,
        max_hops=args.max_hops,
        max_entities=args.max_entities,
        max_triples=args.max_triples,
        require_fully_reviewed=args.require_fully_reviewed,
        reviewed_class_ids=reviewed_ids,
    )
    _save_outputs(
        args.output_dir,
        serialize_graph(graph, mapping_file_sha256=hashlib.sha256(raw).hexdigest()),
        dry_run=args.dry_run,
    )
    print(json.dumps(graph["metadata"], indent=2))
    if not graph["metadata"]["ready_for_gat_kg_training"]:
        print("[DRAFT] Unreviewed/unresolved classes remain. Do NOT train final GAT+KG from this draft.")


if __name__ == "__main__":
    main()
