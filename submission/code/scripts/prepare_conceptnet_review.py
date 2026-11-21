"""Prepare and check a human review of VG150-to-ConceptNet class links.

The generated worksheet is deliberately blank. This script never approves a
lexical match or a ConceptNet assertion on behalf of a human reviewer.
"""

import argparse
import csv
import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path

from scripts.build_conceptnet_graph import write_outputs

REVIEW_FIELDS = (
    "class_id",
    "label",
    "priority",
    "conceptnet_uri_candidates",
    "edge_evidence_count",
    "decision",
    "selected_uri",
    "reviewer",
    "evidence_reference",
    "rationale",
)
EVIDENCE_FIELDS = (
    "class_id",
    "label",
    "candidate_uri",
    "direction",
    "relation",
    "other_uri",
    "assertion_uri",
    "dataset",
    "license",
    "weight",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source(kg_dir: Path) -> tuple[list[dict], dict, list[str], list[str]]:
    metadata = json.loads((kg_dir / "metadata.json").read_text(encoding="utf-8"))
    classes = json.loads((kg_dir / "class_concept_candidates.json").read_text(encoding="utf-8"))["classes"]
    entities = json.loads((kg_dir / "entities.json").read_text(encoding="utf-8"))["entities"]
    relations = json.loads((kg_dir / "relations.json").read_text(encoding="utf-8"))["relations"]
    if (
        metadata.get("source") != "conceptnet"
        or metadata.get("schema_version") != 1
        or metadata.get("ready_for_gat_kg_training") is not False
        or len(classes) != 151
        or any(row.get("class_id") != i for i, row in enumerate(classes))
        or classes[0].get("concept_ids") != []
    ):
        raise ValueError("Expected the 151-row unreviewed ConceptNet candidate graph")
    for row in classes[1:]:
        uris, ids = row.get("concept_uris"), row.get("concept_ids")
        if not isinstance(uris, list) or not isinstance(ids, list) or len(uris) != len(ids):
            raise ValueError(f"Invalid candidate list for class {row['class_id']}")
        if any(
            type(i) is not int or i < 0 or i >= len(entities) or entities[i] != f"cn:{uri}"
            for uri, i in zip(uris, ids, strict=True)
        ):
            raise ValueError(f"Candidate/entity mismatch for class {row['class_id']}")
    return classes, metadata, entities, relations


def _evidence(kg_dir: Path, classes: list[dict], entities: list[str], relations: list[str]) -> tuple[dict, dict]:
    wanted = defaultdict(list)
    for row in classes[1:]:
        for uri in row["concept_uris"]:
            wanted[f"cn:{uri}"].append((row["class_id"], row["label"], uri))
    by_candidate = defaultdict(list)
    all_assertions = defaultdict(set)
    with (kg_dir / "triples.tsv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {"subject_id", "relation_id", "object_id", "assertion_uri", "dataset", "license", "weight"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Invalid ConceptNet evidence columns")
        for edge in reader:
            subject = entities[int(edge["subject_id"])]
            object_ = entities[int(edge["object_id"])]
            relation = relations[int(edge["relation_id"])]
            for concept, direction, other in ((subject, "out", object_), (object_, "in", subject)):
                for class_id, label, uri in wanted.get(concept, ()):
                    key = (class_id, uri)
                    all_assertions[key].add(edge["assertion_uri"])
                    by_candidate[key].append(
                        {
                            "class_id": class_id,
                            "label": label,
                            "candidate_uri": uri,
                            "direction": direction,
                            "relation": relation,
                            "other_uri": other.removeprefix("cn:"),
                            "assertion_uri": edge["assertion_uri"],
                            "dataset": edge["dataset"],
                            "license": edge["license"],
                            "weight": edge["weight"],
                        }
                    )
    return by_candidate, all_assertions


def prepare(kg_dir: Path) -> tuple[dict[str, bytes], dict]:
    classes, metadata, entities, relations = _source(kg_dir)
    by_candidate, _ = _evidence(kg_dir, classes, entities, relations)
    with (kg_dir / "review_queue.csv").open(encoding="utf-8", newline="") as stream:
        original_queue = list(csv.DictReader(stream))
    if len(original_queue) != 150 or any(
        item["class_id"] != str(row["class_id"]) or item["label"] != row["label"]
        for item, row in zip(original_queue, classes[1:], strict=True)
    ):
        raise ValueError("Original candidate review queue does not match VG150 classes")
    worksheet = io.StringIO(newline="")
    review_writer = csv.DictWriter(worksheet, fieldnames=REVIEW_FIELDS, lineterminator="\n")
    review_writer.writeheader()
    samples = io.StringIO(newline="")
    evidence_writer = csv.DictWriter(samples, fieldnames=EVIDENCE_FIELDS, lineterminator="\n")
    evidence_writer.writeheader()
    priority_counts = defaultdict(int)
    for row, original in zip(classes[1:], original_queue, strict=True):
        candidates = row["concept_uris"]
        # Prior WordNet ambiguity is only a review-order hint. It never
        # determines the ConceptNet decision or selects an URI.
        priority = (
            "ambiguous_label"
            if original["wordnet_status"] == "ambiguous"
            else "multiple_candidate_uris"
            if len(candidates) > 1
            else "standard"
        )
        priority_counts[priority] += 1
        review_writer.writerow(
            {
                "class_id": row["class_id"],
                "label": row["label"],
                "priority": priority,
                "conceptnet_uri_candidates": ";".join(candidates),
                "edge_evidence_count": sum(len(by_candidate[row["class_id"], uri]) for uri in candidates),
            }
        )
        for uri in candidates:
            edges = by_candidate[row["class_id"], uri]
            # One reproducible, high-weight example per relation and direction.
            # This is a navigation aid, never a claim that all edges are sound.
            edges.sort(key=lambda e: (-float(e["weight"]), e["assertion_uri"], e["other_uri"]))
            picked = {}
            for edge in edges:
                picked.setdefault((edge["relation"], edge["direction"]), edge)
            for edge in sorted(picked.values(), key=lambda e: (e["relation"], e["direction"])):
                evidence_writer.writerow(edge)
    source_hashes = {
        name: _sha256(kg_dir / name)
        for name in (
            "entities.json",
            "relations.json",
            "triples.tsv",
            "class_concept_candidates.json",
            "metadata.json",
            "review_queue.csv",
        )
    }
    provenance = {
        "schema_version": 1,
        "purpose": "human_review_worksheet_not_training_approval",
        "conceptnet_source_version": metadata["source_version"],
        "conceptnet_snapshot_sha256": metadata["source_snapshot_sha256"],
        "source_hashes": source_hashes,
        "class_count": 150,
        "priority_counts": dict(sorted(priority_counts.items())),
        "review_status": "unreviewed",
    }
    return {
        "review.csv": worksheet.getvalue().encode("utf-8"),
        "evidence_samples.csv": samples.getvalue().encode("utf-8"),
        "provenance.json": (json.dumps(provenance, indent=2) + "\n").encode("utf-8"),
    }, provenance


def check(kg_dir: Path, review_dir: Path, *, require_complete: bool = False) -> dict:
    classes, _, entities, relations = _source(kg_dir)
    provenance = json.loads((review_dir / "provenance.json").read_text(encoding="utf-8"))
    for name, expected in provenance["source_hashes"].items():
        if _sha256(kg_dir / name) != expected:
            raise ValueError(f"Review worksheet/source hash mismatch: {name}")
    _, all_assertions = _evidence(kg_dir, classes, entities, relations)
    with (review_dir / "review.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(REVIEW_FIELDS):
            raise ValueError("Review worksheet columns or order changed")
        rows = list(reader)
    if len(rows) != 150:
        raise ValueError("Review worksheet must contain exactly 150 classes")
    counts = {"accept": 0, "reject": 0, "defer": 0, "pending": 0}
    for expected, row in zip(classes[1:], rows, strict=True):
        class_id = expected["class_id"]
        candidates = expected["concept_uris"]
        if (
            row["class_id"] != str(class_id)
            or row["label"] != expected["label"]
            or row["conceptnet_uri_candidates"] != ";".join(candidates)
        ):
            raise ValueError(f"Class ID/label/candidate mismatch at row {class_id}")
        decision = row["decision"].strip().lower()
        if not decision:
            if any(row[k].strip() for k in ("selected_uri", "reviewer", "evidence_reference", "rationale")):
                raise ValueError(f"Partial review without decision at class {class_id}")
            counts["pending"] += 1
            continue
        if decision not in ("accept", "reject", "defer"):
            raise ValueError(f"Invalid decision at class {class_id}")
        selected = row["selected_uri"].strip()
        if decision == "accept" and selected not in candidates:
            raise ValueError(f"Accepted URI must be one listed candidate at class {class_id}")
        if decision != "accept" and selected:
            raise ValueError(f"Only accepted classes can select a URI at class {class_id}")
        reviewer = row["reviewer"].strip()
        if not reviewer or reviewer.casefold() in {"ai", "openai", "chatgpt", "assistant", "llm"}:
            raise ValueError(f"Decision needs a named human reviewer at class {class_id}")
        if not row["rationale"].strip():
            raise ValueError(f"Decision needs a rationale at class {class_id}")
        evidence = row["evidence_reference"].strip()
        if decision == "accept" and evidence not in all_assertions[class_id, selected]:
            raise ValueError(f"Accepted class needs one matching assertion URI at class {class_id}")
        if decision != "accept" and not evidence:
            raise ValueError(f"Decision needs an evidence reference at class {class_id}")
        counts[decision] += 1
    if require_complete and counts["pending"]:
        raise ValueError(f"ConceptNet review incomplete: {counts['pending']} pending classes")
    return counts


def compile_reviewed(kg_dir: Path, review_dir: Path, output_dir: Path) -> dict[str, bytes]:
    """Bind complete human class decisions to a separate KG version.

    Source assertions are copied byte-for-byte. The approved scope is the
    class-to-concept mapping, not every noisy ConceptNet assertion.
    """
    counts = check(kg_dir, review_dir, require_complete=True)
    if counts["accept"] == 0:
        raise ValueError("Cannot compile a reviewed KG with zero accepted class links")
    review_provenance = json.loads((review_dir / "provenance.json").read_text(encoding="utf-8"))
    review_basis = review_provenance.get("review_basis", "individual_conceptnet_class_uri_selection")
    if review_basis not in {
        "individual_conceptnet_class_uri_selection",
        "human_batch_approved_ai_proposal",
        "human_mixed_direct_and_ai_batch",
    }:
        raise ValueError("Unsupported ConceptNet review basis")
    if review_basis == "human_mixed_direct_and_ai_batch" and (
        review_provenance.get("human_direct_class_count") != 16
        or review_provenance.get("human_batch_approved_class_count") != 134
    ):
        raise ValueError("Mixed ConceptNet review provenance must describe 16 direct and 134 batch-approved rows")
    classes, source_metadata, entities, _ = _source(kg_dir)
    with (review_dir / "review.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    entity_lookup = {name: index for index, name in enumerate(entities)}
    class_ids = [None]
    links = [{"class_id": 0, "label": "__background__", "decision": "background", "selected_uri": None}]
    reviews = []
    for candidate, row in zip(classes[1:], rows, strict=True):
        uri = row["selected_uri"].strip() if row["decision"].strip().lower() == "accept" else None
        class_ids.append(entity_lookup[f"cn:{uri}"] if uri else None)
        links.append(
            {
                "class_id": candidate["class_id"],
                "label": candidate["label"],
                "decision": row["decision"].strip().lower(),
                "selected_uri": uri,
            }
        )
        reviews.append(
            {
                "class_id": candidate["class_id"],
                "label": candidate["label"],
                "decision": row["decision"].strip().lower(),
                "selected_uri": uri,
                "reviewer": row["reviewer"].strip(),
                "evidence_reference": row["evidence_reference"].strip(),
                "rationale": row["rationale"].strip(),
            }
        )
    source_hashes = {
        name: _sha256(kg_dir / name)
        for name in ("entities.json", "relations.json", "triples.tsv", "class_concept_candidates.json", "metadata.json")
    }
    metadata = {
        **source_metadata,
        "curation_tier": (
            "HUMAN_MIXED_DIRECT_AND_BATCH_APPROVED_AI_CLASS_LINKS_SOURCE_ASSERTIONS_UNREVIEWED"
            if review_basis == "human_mixed_direct_and_ai_batch"
            else "HUMAN_BATCH_APPROVED_AI_CLASS_LINKS_SOURCE_ASSERTIONS_UNREVIEWED"
            if review_basis == "human_batch_approved_ai_proposal"
            else "HUMAN_REVIEWED_CLASS_LINKS_SOURCE_ASSERTIONS_UNREVIEWED"
        ),
        "human_verified_class_count": (
            16
            if review_basis == "human_mixed_direct_and_ai_batch"
            else 0
            if review_basis == "human_batch_approved_ai_proposal"
            else 150
        ),
        "human_batch_approved_class_count": (
            134
            if review_basis == "human_mixed_direct_and_ai_batch"
            else 150
            if review_basis == "human_batch_approved_ai_proposal"
            else 0
        ),
        "resolved_class_count": counts["accept"],
        "rejected_class_count": counts["reject"],
        "deferred_class_count": counts["defer"],
        "unresolved_class_ids": [r["class_id"] for r in reviews if r["decision"] != "accept"],
        "ready_for_gat_kg_training": True,
        "review_scope": "class_to_concept_uri_only_assertions_source_filtered_not_individually_reviewed",
        "review_basis": review_basis,
        "input_review_sha256": _sha256(review_dir / "review.csv"),
        "source_graph_sha256": source_hashes,
    }

    def to_json(data):
        return (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode("utf-8")

    outputs = {
        name: (kg_dir / name).read_bytes()
        for name in ("entities.json", "relations.json", "triples.tsv", "class_concept_candidates.json")
    }
    outputs.update(
        {
            "metadata.json": to_json(metadata),
            "class_entity_ids.json": to_json({"class_entity_ids": class_ids}),
            "class_concept_links.json": to_json({"classes": links}),
            "review.csv": (review_dir / "review.csv").read_bytes(),
            "provenance.json": (review_dir / "provenance.json").read_bytes(),
        }
    )
    if review_basis == "human_batch_approved_ai_proposal":
        for name, key in (
            ("approval.json", "approval_sha256"),
            ("review_suggestions.ai-draft.csv", "proposal_sha256"),
            ("review_focus.ai-draft.csv", "focus_sha256"),
        ):
            data = (review_dir / name).read_bytes()
            if hashlib.sha256(data).hexdigest() != review_provenance.get(key):
                raise ValueError(f"Batch approval provenance mismatch: {name}")
            outputs[name] = data
    if review_basis == "human_mixed_direct_and_ai_batch":
        for name, key in (
            ("approval.json", "approval_sha256"),
            ("selection.csv", "selection_sha256"),
            ("direct_decisions.csv", "direct_decisions_sha256"),
            ("review_suggestions.ai-draft.csv", "proposal_sha256"),
            ("review_focus.ai-draft.csv", "focus_sha256"),
        ):
            data = (review_dir / name).read_bytes()
            if hashlib.sha256(data).hexdigest() != review_provenance.get(key):
                raise ValueError(f"Mixed review provenance mismatch: {name}")
            outputs[name] = data
    manifest = {
        "schema_version": 1,
        "kg_source": "conceptnet",
        "kg_path": str(output_dir.resolve()),
        "kg_sha256": {
            name: hashlib.sha256(outputs[name]).hexdigest()
            for name in ("entities.json", "relations.json", "triples.tsv", "metadata.json")
        },
        "review_status": "human_approved",
        "review_basis": review_basis,
        "review_scope": metadata["review_scope"],
        "review_csv_sha256": hashlib.sha256(outputs["review.csv"]).hexdigest(),
        "class_concept_candidates_sha256": hashlib.sha256(outputs["class_concept_candidates.json"]).hexdigest(),
        "provenance_sha256": hashlib.sha256(outputs["provenance.json"]).hexdigest(),
        "class_entity_ids_sha256": hashlib.sha256(outputs["class_entity_ids.json"]).hexdigest(),
        "class_concept_links_sha256": hashlib.sha256(outputs["class_concept_links.json"]).hexdigest(),
        "reviews": reviews,
    }
    if review_basis == "human_batch_approved_ai_proposal":
        manifest.update(
            approval_sha256=review_provenance["approval_sha256"],
            proposal_sha256=review_provenance["proposal_sha256"],
            focus_sha256=review_provenance["focus_sha256"],
            approved_by=review_provenance["approved_by"],
        )
    if review_basis == "human_mixed_direct_and_ai_batch":
        manifest.update(
            approval_sha256=review_provenance["approval_sha256"],
            proposal_sha256=review_provenance["proposal_sha256"],
            focus_sha256=review_provenance["focus_sha256"],
            selection_sha256=review_provenance["selection_sha256"],
            direct_decisions_sha256=review_provenance["direct_decisions_sha256"],
            approved_by=review_provenance["approved_by"],
            human_direct_class_count=16,
            human_batch_approved_class_count=134,
        )
    outputs["review_manifest.json"] = to_json(manifest)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "check", "compile"))
    parser.add_argument("--kg-dir", type=Path, default=Path("data/knowledge/conceptnet-b2-v1"))
    parser.add_argument("--review-dir", type=Path, default=Path("data/knowledge/conceptnet-review-v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/knowledge/conceptnet-reviewed-v1"))
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.action == "prepare":
        outputs, info = prepare(args.kg_dir)
        write_outputs(args.review_dir, outputs, dry_run=args.dry_run)
        print(json.dumps(info, indent=2))
    elif args.action == "check":
        print(json.dumps(check(args.kg_dir, args.review_dir, require_complete=args.require_complete), indent=2))
    else:
        outputs = compile_reviewed(args.kg_dir, args.review_dir, args.output_dir)
        write_outputs(args.output_dir, outputs, dry_run=args.dry_run)
        print(f"[CONCEPTNET REVIEWED] {len(outputs)} artifacts prepared under {args.output_dir}")


if __name__ == "__main__":
    main()
