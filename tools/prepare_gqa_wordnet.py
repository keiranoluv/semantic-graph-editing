"""Build a separate, conservative GQA WordNet mapping and a review packet.

Only the train-derived vocabulary is read from the GQA manifest. Existing
VG150 decisions may transfer by exact label, never by class ID or embedding.
Transferred decisions remain subject to review in the GQA context.
"""

import argparse
import csv
import io
import json
import random
from pathlib import Path

from scripts.build_wordnet_graph import serialize_graph
from scripts.prepare_knowledge import _save_outputs
from src.knowledge.entity_linking import link_vocabulary
from src.knowledge.graph_builder import build_wordnet_graph
from src.knowledge.transe.core import check_gate, load_kg, sha256


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def csv_bytes(rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def transfer_overrides(vocabulary, source_mapping, source_overrides):
    """Reindex explicitly curated lexical decisions; never infer plural senses."""
    decisions = {}
    provenance = {}
    for row in source_mapping["rows"][1:]:
        label = row["label"]
        if label not in vocabulary["label_to_idx"] or row["status"] != "linked_manual":
            continue
        audit = source_overrides.get("review_provenance", {}).get(label, {})
        if source_overrides["mappings"].get(label) != row["synset_id"] or not audit.get("reviewer"):
            raise ValueError(f"Missing source review evidence for {label}")
        decisions[label] = row["synset_id"]
        provenance[label] = {
            "source_class_id": row["class_id"],
            "gqa_class_id": vocabulary["label_to_idx"][label],
            "source_reviewer": audit["reviewer"],
            "basis": "exact_label_transfer_from_VG150_not_GQA_context_review",
        }
    return {"version": 1, "mappings": decisions, "transfer_provenance": provenance}


def review_packet(mapping, *, sample_size=30, seed=42):
    """Stratified deterministic sample, plus a complete editable lexical audit."""
    if sample_size < 1:
        raise ValueError("Review sample size must be positive")
    rows = []
    for row in mapping["rows"][1:]:
        rows.append(
            {
                "class_id": row["class_id"],
                "label": row["label"],
                "status": row["status"],
                "synset_id": row["synset_id"] or "",
                "definition": next(
                    (c["definition"] for c in row["candidates"] if c["synset_id"] == row["synset_id"]), ""
                ),
                "candidates": " | ".join(f"{c['synset_id']}: {c['definition']}" for c in row["candidates"]),
                "decision": "",
                "selected_synset": "",
                "reviewer": "",
                "notes": "",
            }
        )
    rng = random.Random(seed)
    buckets = {}
    for row in rows:
        buckets.setdefault(row["status"], []).append(row)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    sample = []
    while len(sample) < min(sample_size, len(rows)):
        for status in sorted(buckets):
            if buckets[status] and len(sample) < sample_size:
                sample.append(buckets[status].pop())
    return rows, sorted(sample, key=lambda r: r["class_id"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/level2/gqa-v1/manifest.json"))
    parser.add_argument("--source-kg", type=Path, default=Path("data/knowledge/wordnet-kg-toan-v1"))
    parser.add_argument("--output", type=Path, default=Path("data/knowledge/gqa-wordnet-v1"))
    parser.add_argument("--report", type=Path, default=Path("reports/level2/gqa-wordnet-v1"))
    args = parser.parse_args(argv)
    import nltk
    from nltk.corpus import wordnet

    wordnet.ensure_loaded()
    source_review = args.source_kg / "review_manifest.json"
    check_gate("reviewed", load_kg(args.source_kg), args.source_kg, source_review)
    review = json.loads(source_review.read_text())
    source_mapping_path, source_overrides_path = [Path(review[f"{k}_path"]) for k in ("mapping", "overrides")]
    source_mapping = json.loads(source_mapping_path.read_text())
    if source_mapping["source_version"] != wordnet.get_version():
        raise ValueError("Source review and local WordNet versions differ")
    manifest = json.loads(args.manifest.read_text())
    vocabulary = {"label_to_idx": manifest["vocabulary"]["label_to_idx"]}
    overrides = transfer_overrides(vocabulary, source_mapping, json.loads(source_overrides_path.read_text()))
    _save_outputs(
        args.output,
        {"vocabulary.json": json_bytes(vocabulary), "transferred_overrides.json": json_bytes(overrides)},
        dry_run=False,
    )
    mapping, coverage = link_vocabulary(
        args.output / "vocabulary.json",
        wordnet=wordnet,
        expected_classes=len(vocabulary["label_to_idx"]),
        overrides_path=args.output / "transferred_overrides.json",
        nltk_version=nltk.__version__,
    )
    all_rows, sample = review_packet(mapping)
    corpus_path = Path(str(wordnet.root.zipfile.filename)) if hasattr(wordnet.root, "zipfile") else None
    sources = {
        "gqa_manifest_sha256": sha256(args.manifest),
        "source_review_manifest_sha256": sha256(source_review),
        "source_mapping_sha256": sha256(source_mapping_path),
        "source_overrides_sha256": sha256(source_overrides_path),
        "wordnet_version": wordnet.get_version(),
        "wordnet_zip_sha256": sha256(corpus_path) if corpus_path is not None else None,
        "source_code_sha256": {
            name: sha256(Path(name))
            for name in (
                "tools/prepare_gqa_wordnet.py",
                "src/knowledge/entity_linking.py",
                "src/knowledge/graph_builder.py",
                "src/knowledge/sense_review.py",
                "scripts/build_wordnet_graph.py",
                "scripts/prepare_knowledge.py",
            )
        },
    }
    coverage.update(
        {
            "scope": "GQA_mapping_preparation_not_final_experiment",
            "sources": sources,
            "gqa_context_human_review_completed": False,
            "source_review_transfer_count": len(overrides["mappings"]),
            "review_sample_size": len(sample),
            "review_sample_seed": 42,
            "review_sample_policy": "round_robin_status_strata_seeded_without_replacement_not_population_accuracy",
            "missing_policy": "zero_vector_and_false_valid_bit",
            "test_annotation_opened": False,
            "scene_relation_annotations_read": False,
            "note": "Single noun sense is lexical uniqueness, not verified visual meaning. "
            "VG150 decisions transfer by exact label only; no new GQA human review is claimed.",
        }
    )
    _save_outputs(
        args.output, {"wordnet_mapping.json": json_bytes(mapping), "coverage.json": json_bytes(coverage)}, dry_run=False
    )
    graph = build_wordnet_graph(mapping, wordnet=wordnet, max_hops=2, max_entities=30000, max_triples=100000)
    _save_outputs(
        args.output / "kg",
        serialize_graph(graph, mapping_file_sha256=sha256(args.output / "wordnet_mapping.json")),
        dry_run=False,
    )
    report = {**coverage, "kg": graph["metadata"], "mapping_sha256": sha256(args.output / "wordnet_mapping.json")}
    _save_outputs(
        args.report,
        {
            "coverage.json": json_bytes(report),
            "review_all.csv": csv_bytes(all_rows, list(all_rows[0])),
            "review_sample.csv": csv_bytes(sample, list(all_rows[0])),
        },
        dry_run=False,
    )
    print(json.dumps({k: v for k, v in report.items() if k not in ("sources", "kg")}, indent=2))
    print(f"KG: {len(graph['entities'])} entities / {len(graph['triples'])} external WordNet triples")


if __name__ == "__main__":
    main()
