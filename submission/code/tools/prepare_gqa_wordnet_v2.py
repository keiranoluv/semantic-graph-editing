"""Apply the explicit GQA sample review, audit all train objects, build WordNet v2."""

import argparse
import copy
import csv
import io
import json
from collections import Counter
from pathlib import Path

from scripts.build_wordnet_graph import serialize_graph
from scripts.prepare_knowledge import _save_outputs
from src.knowledge.graph_builder import build_wordnet_graph
from src.knowledge.sense_review import _validate_mapping
from src.knowledge.transe.core import sha256
from tools.apply_gqa_review_snapshot import validate_review
from tools.prepare_gqa_wordnet import json_bytes
from tools.prepare_gqa_wordnet_review import canonical_hash, train_records


def apply_review(base, decisions, *, provenance):
    """Only explicit reviewed IDs change; defer removes links without inventing senses."""
    result = copy.deepcopy(base)
    rows = _validate_mapping(result)
    seen, changes = set(), []
    for entry in decisions:
        i = int(entry["class_id"])
        if i in seen or not 0 < i < len(rows) or entry["label"] != rows[i]["label"]:
            raise ValueError("Invalid or repeated reviewed class ID/label")
        seen.add(i)
        row = rows[i]
        before = row["synset_id"]
        if not entry["reviewer"].strip():
            raise ValueError("Explicit reviewer name required")
        if entry["decision"] == "accept":
            if entry["selected_synset"] not in {c["synset_id"] for c in row["candidates"]}:
                raise ValueError("Accepted synset is not a pinned candidate")
            row.update(status="linked_manual", synset_id=entry["selected_synset"], match_rule="gqa_human_sample_review")
        elif entry["decision"] == "defer":
            if entry["selected_synset"]:
                raise ValueError("Deferred links must have no selected synset")
            row.update(
                status="ambiguous" if row["candidates"] else "unmatched",
                synset_id=None,
                match_rule="gqa_human_explicit_defer",
            )
        else:
            raise ValueError("Every sample row requires accept/defer")
        row["gqa_human_review"] = {
            "decision": entry["decision"],
            "reviewer": entry["reviewer"],
            "notes": entry["notes"],
            **provenance,
        }
        if before != row["synset_id"]:
            changes.append({"class_id": i, "label": row["label"], "before": before, "after": row["synset_id"]})
    result["gqa_review"] = {
        "scope": "explicit_stratified_sample_only",
        "class_ids": sorted(seen),
        "accept_count": sum(e["decision"] == "accept" for e in decisions),
        "defer_count": sum(e["decision"] == "defer" for e in decisions),
        "all_classes_human_reviewed": len(seen) == len(rows) - 1,
        **provenance,
    }
    return result, changes


def audit_objects(records, mapping, base):
    counts, images, total_images = Counter(), Counter(), 0
    for record in records:
        total_images += 1
        ids = [o["label_id"] for o in record["objects"]]
        if any(not 0 < i < len(mapping["rows"]) for i in ids):
            raise ValueError("Train object label outside pinned vocabulary")
        counts.update(ids)
        images.update(set(ids))
    linked = {r["class_id"] for r in mapping["rows"][1:] if r["synset_id"] is not None}
    old_linked = {r["class_id"] for r in base["rows"][1:] if r["synset_id"] is not None}
    objects = sum(counts.values())
    if not objects:
        raise ValueError("No train objects to audit")
    rows = [
        {
            "class_id": r["class_id"],
            "label": r["label"],
            "status": r["status"],
            "synset_id": r["synset_id"] or "",
            "train_objects": counts[r["class_id"]],
            "train_images": images[r["class_id"]],
            "linked": r["class_id"] in linked,
            "gqa_review_decision": r.get("gqa_human_review", {}).get("decision", ""),
        }
        for r in mapping["rows"][1:]
    ]
    return {
        "train_images": total_images,
        "train_objects": objects,
        "linked_classes": len(linked),
        "class_count": len(rows),
        "class_coverage": len(linked) / len(rows),
        "linked_objects": sum(counts[i] for i in linked),
        "object_coverage": sum(counts[i] for i in linked) / objects,
        "v1_linked_objects": sum(counts[i] for i in old_linked),
        "v1_object_coverage": sum(counts[i] for i in old_linked) / objects,
        "per_class": rows,
        "unlinked_by_train_frequency": sorted(
            (r for r in rows if not r["linked"]), key=lambda r: (-r["train_objects"], r["class_id"])
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("data/knowledge/gqa-wordnet-v1"))
    parser.add_argument(
        "--csv", type=Path, default=Path("reviews/gqa-wordnet-human-v1/submissions/review-ce1dcb634ff05050.csv")
    )
    parser.add_argument("--packet", type=Path, default=Path("reviews/gqa-wordnet-human-v1/review-data.json"))
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/level2/gqa-v1/manifest.json"))
    parser.add_argument("--cache", type=Path, default=Path("data/cache/gqa-full-clip-v1"))
    parser.add_argument("--output", type=Path, default=Path("data/knowledge/gqa-wordnet-v2"))
    parser.add_argument("--report", type=Path, default=Path("reports/level2/gqa-wordnet-v2"))
    args = parser.parse_args(argv)
    from nltk.corpus import wordnet

    wordnet.ensure_loaded()
    packet = json.loads(args.packet.read_text())
    if canonical_hash({k: v for k, v in packet.items() if k != "packet_sha256"}) != packet["packet_sha256"]:
        raise ValueError("Review packet hash mismatch")
    base_path = args.base / "wordnet_mapping.json"
    if (
        sha256(base_path) != packet["provenance"]["mapping_sha256"]
        or sha256(args.manifest) != packet["provenance"]["gqa_manifest_sha256"]
    ):
        raise ValueError("Reviewed mapping/manifest changed")
    decisions = validate_review(packet, args.csv)
    base = json.loads(base_path.read_text())
    provenance = {"csv_sha256": sha256(args.csv), "review_packet_sha256": packet["packet_sha256"]}
    mapping, changes = apply_review(base, decisions, provenance=provenance)
    if mapping["source_version"] != wordnet.get_version():
        raise ValueError("WordNet version differs from reviewed source")
    for row in mapping["rows"][1:]:
        if row["synset_id"] is not None:
            sense = wordnet.synset(row["synset_id"])
            if sense.pos() != "n" or sense.name() != row["synset_id"]:
                raise ValueError("Invalid mapped noun synset")
    manifest = json.loads(args.manifest.read_text())
    labels = manifest["vocabulary"]["label_to_idx"]
    if any(mapping["rows"][i]["label"] != label for label, i in labels.items()):
        raise ValueError("Mapping label IDs do not match GQA vocabulary")
    read_provenance = {}
    audit = audit_objects(train_records(args.cache, manifest, args.manifest, read_provenance), mapping, base)
    if audit["train_images"] != len(manifest["splits"]["train"]):
        raise ValueError("Incomplete full-train audit")
    coverage = {
        **{k: v for k, v in audit.items() if k not in ("per_class", "unlinked_by_train_frequency")},
        "sources": {
            "gqa_manifest_sha256": sha256(args.manifest),
            "base_mapping_sha256": sha256(base_path),
            **provenance,
            "source_sha256": {str(Path(__file__)): sha256(Path(__file__))},
        },
        "gqa_context_human_review_completed": False,
        "gqa_sample_review_completed": True,
        "reviewed_sample": mapping["gqa_review"],
        "changes": changes,
        "interpretation": "30 reviewed labels, not review of all 1699 classes or population accuracy",
        "test_annotation_opened": False,
        "validation_shards_opened": False,
    }
    _save_outputs(
        args.output,
        {
            "wordnet_mapping.json": json_bytes(mapping),
            "coverage.json": json_bytes(coverage),
            "vocabulary.json": (args.base / "vocabulary.json").read_bytes(),
        },
        dry_run=False,
    )
    graph = build_wordnet_graph(
        mapping,
        wordnet=wordnet,
        max_hops=2,
        max_entities=30000,
        max_triples=100000,
        reviewed_class_ids=frozenset(int(e["class_id"]) for e in decisions if e["decision"] == "accept"),
    )
    _save_outputs(
        args.output / "kg",
        serialize_graph(graph, mapping_file_sha256=sha256(args.output / "wordnet_mapping.json")),
        dry_run=False,
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(audit["per_class"][0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(audit["per_class"])
    _save_outputs(
        args.report,
        {
            "coverage.json": json_bytes({**coverage, "kg": graph["metadata"]}),
            "object-frequency.csv": stream.getvalue().encode(),
            "unlinked-priority.json": json_bytes(audit["unlinked_by_train_frequency"]),
            "train-read-provenance.json": json_bytes(read_provenance),
        },
        dry_run=False,
    )
    print(
        json.dumps(
            {
                **{k: v for k, v in coverage.items() if k not in ("sources", "reviewed_sample", "changes")},
                "entities": len(graph["entities"]),
                "triples": len(graph["triples"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
