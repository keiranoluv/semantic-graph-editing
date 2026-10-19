"""Create an AI-provisional WordNet mapping and KG, NEVER an audited mapping.

This intentionally bypasses the human-only ``prepare_wordnet_review compile``
command. Inputs are pinned to the original A.1 mapping and its A.2 ranking.
No Visual Genome images, annotations, or predicate labels are read.

Run with ``python -m scripts.build_ai_wordnet_draft --help``.
"""

import argparse
import copy
import csv
import hashlib
import io
import json
from pathlib import Path

from scripts.build_wordnet_graph import serialize_graph
from scripts.prepare_knowledge import _save_outputs
from src.knowledge.graph_builder import build_wordnet_graph
from src.knowledge.sense_review import REVIEW_FIELDS, _sha256_json, _validate_mapping

AI_REVIEWER = "ChatGPT-AI-provisional-not-human-verified"
DRAFT_LABEL = "AI_PROVISIONAL_NOT_HUMAN_VERIFIED"


def _json_bytes(obj: dict) -> bytes:
    return (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def prepare_draft(mapping: dict, ranking: dict, review_bytes: bytes) -> tuple[dict, dict, bytes]:
    """Convert explicitly AI-labeled suggestions to a NON-AUDITED B.1 input.

    ``linked_manual`` is used only as the legacy graph builder's representation
    of a chosen synset. Each AI-chosen row is separately tagged as provisional;
    it MUST NOT be passed to the human-review gate or presented as verified.
    """
    base_rows = _validate_mapping(mapping)
    if len(base_rows) != 151:
        raise ValueError("Expected 151 class rows including background (VG150)")
    if ranking.get("schema_version") != 1 or ranking.get("source_version") != mapping.get("source_version"):
        raise ValueError("Ranking schema or WordNet version differs from A.1 mapping")
    if ranking.get("mapping_sha256") != _sha256_json(mapping):
        raise ValueError("Ranking does not match source mapping SHA256")
    ranked_rows = ranking.get("rows")
    if not isinstance(ranked_rows, list) or len(ranked_rows) != 150:
        raise ValueError("Ranking must contain 150 classes")

    try:
        reader = csv.DictReader(io.StringIO(review_bytes.decode("utf-8-sig"), newline=""))
        if not set(REVIEW_FIELDS).issubset(reader.fieldnames or []):
            raise ValueError("Review CSV columns changed")
        choices = list(reader)
    except UnicodeDecodeError as exc:
        raise ValueError("Review CSV must be UTF-8") from exc
    if len(choices) != 150:
        raise ValueError("Review CSV must contain exactly 150 class rows")

    draft = copy.deepcopy(mapping)
    draft["overrides_sha256"] = None  # Never claim a matching human-reviewed overrides file.
    draft["curation_tier"] = DRAFT_LABEL
    draft["human_verified_class_count"] = 0
    draft["source_mapping_canonical_sha256"] = _sha256_json(mapping)
    draft["source_review_csv_sha256"] = hashlib.sha256(review_bytes).hexdigest()
    draft["source_ranking_canonical_sha256"] = _sha256_json(ranking)
    draft["curation_warning"] = "AI lexical suggestions only; not reviewed against VG images."

    accepted_ids = []
    new_ids = []
    original_auto_ids = []
    deferred_ids = []
    review = io.StringIO(newline="")
    queue = csv.writer(review, lineterminator="\n")
    queue.writerow(("class_id", "label", "suggested_synset", "candidate_definitions", "reason"))
    for class_row, ranked, item, out_row in zip(base_rows[1:], ranked_rows, choices, draft["rows"][1:], strict=True):
        class_id = class_row["class_id"]
        identity = (str(class_id), class_row["label"], class_row["status"])
        if (item["class_id"], item["label"], item["status"]) != identity:
            raise ValueError(f"Review CSV alignment mismatch at class {class_id}")
        if (ranked.get("class_id"), ranked.get("label"), ranked.get("status")) != (
            class_id,
            class_row["label"],
            class_row["status"],
        ):
            raise ValueError(f"Ranking alignment mismatch at class {class_id}")
        candidates = {c["synset_id"] for c in class_row["candidates"]}
        if {c["synset_id"] for c in ranked["ranked_candidates"]} != candidates:
            raise ValueError(f"Ranking candidate set mismatch at class {class_id}")
        decision = (item["decision"] or "").strip().casefold()
        selected = (item["selected_synset"] or "").strip()
        reviewer = (item["reviewer"] or "").strip()
        notes = (item["notes"] or "").strip()
        if decision not in ("accept", "defer"):
            raise ValueError(f"Only explicit accept/defer are allowed for AI draft at class {class_id}")
        if decision == "accept":
            if reviewer != AI_REVIEWER or not notes or "provisional" not in notes.casefold():
                raise ValueError(f"AI source declaration/rationale missing at class {class_id}")
            if not selected or selected not in candidates:
                raise ValueError(f"Accepted choice not in original WordNet candidates at class {class_id}")
            if class_row["status"] == "linked_auto":
                if selected != class_row["synset_id"]:
                    raise ValueError(f"Auto-linked sense may not silently change at class {class_id}")
                original_auto_ids.append(class_id)
            else:
                if class_row["status"] not in ("ambiguous", "unmatched"):
                    raise ValueError(f"Unexpected original status at class {class_id}")
                out_row["status"] = "linked_manual"  # Legacy B.1 schema, NOT human approval.
                out_row["synset_id"] = selected
                out_row["match_rule"] = "ai_provisional_choice"
                new_ids.append(class_id)
            out_row["curation_tier"] = DRAFT_LABEL
            out_row["verified_by_human"] = False
            accepted_ids.append(class_id)
        else:
            if selected or reviewer:
                raise ValueError(f"Deferred row must not contain a selection/reviewer at class {class_id}")
            if class_row["status"] == "linked_auto":
                raise ValueError(f"Auto-linked class {class_id} needs an explicit AI audit or separate human review")
            deferred_ids.append(class_id)
            definitions = "; ".join(f"{c['synset_id']}: {c['definition']}" for c in ranked["ranked_candidates"][:4])
            queue.writerow((class_id, class_row["label"], item["suggested_synset"], definitions, notes))
    report = {
        "curation_tier": DRAFT_LABEL,
        "class_count": 150,
        "original_auto_count": len(original_auto_ids),
        "new_ai_provisional_count": len(new_ids),
        "ai_provisional_class_count": len(accepted_ids),
        "deferred_class_count": len(deferred_ids),
        "deferred_class_ids": deferred_ids,
        "candidate_mapping_coverage": len(accepted_ids) / 150,
        "human_verified_class_count": 0,
        "ready_for_gat_kg_training": False,
        "source_mapping_sha256": _sha256_json(mapping),
        "source_ranking_sha256": _sha256_json(ranking),
        "source_review_csv_sha256": hashlib.sha256(review_bytes).hexdigest(),
        "warning": "Compatibility status linked_manual does NOT mean human review. For draft analysis only.",
    }
    return draft, report, review.getvalue().encode("utf-8")


def generate_outputs(
    original: dict,
    ranking: dict,
    review_bytes: bytes,
    *,
    wordnet: object,
    max_hops: int = 2,
    max_entities: int = 10000,
    max_triples: int = 50000,
) -> tuple[dict[str, bytes], dict[str, bytes], dict]:
    draft, report, queue = prepare_draft(original, ranking, review_bytes)
    if str(wordnet.get_version()) != draft["source_version"]:
        raise ValueError("Installed WordNet version does not match pinned A.1 mapping")
    draft_bytes = _json_bytes(draft)
    graph = build_wordnet_graph(
        draft,
        wordnet=wordnet,
        max_hops=max_hops,
        max_entities=max_entities,
        max_triples=max_triples,
        reviewed_class_ids=frozenset(),  # Critical: AI recommendations are NOT human audited.
    )
    graph["metadata"].update(
        {
            "curation_tier": DRAFT_LABEL,
            "human_verified_class_count": 0,
            "ai_provisional_class_count": report["ai_provisional_class_count"],
            "source_mapping_sha256": report["source_mapping_sha256"],
            "source_review_csv_sha256": report["source_review_csv_sha256"],
        }
    )
    if graph["metadata"]["ready_for_gat_kg_training"]:
        raise AssertionError("An AI-generated knowledge graph can never pass the human audit gate")
    mapping_outputs = {
        "wordnet_mapping.json": draft_bytes,
        "ai_draft_summary.json": _json_bytes(report),
        "deferred_review.csv": queue,
    }
    graph_outputs = serialize_graph(graph, mapping_file_sha256=hashlib.sha256(draft_bytes).hexdigest())
    return mapping_outputs, graph_outputs, graph["metadata"]


def _ensure_safe_outputs(root: Path, files_by_dir: dict[Path, dict[str, bytes]]) -> None:
    for path in (root, *files_by_dir):
        if path.is_symlink():
            raise FileExistsError(f"Refusing symlink directory: {path}")
    # Validate all destinations before writing to avoid partial mismatch updates.
    for directory, files in files_by_dir.items():
        for name, data in files.items():
            dest = directory / name
            if dest.is_symlink() or (dest.exists() and dest.read_bytes() != data):
                raise FileExistsError(f"Refusing to replace different file: {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, default=Path("data/knowledge/wordnet-vg150/wordnet_mapping.json"))
    parser.add_argument("--ranking", type=Path, default=Path("data/knowledge/wordnet-review-v1/wordnet_rankings.json"))
    parser.add_argument(
        "--review-csv",
        type=Path,
        default=Path("data/knowledge/wordnet-review-v1/wordnet_review_ai_provisional.csv"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("data/knowledge/wordnet-ai-draft-v1"))
    parser.add_argument("--max-hops", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--max-entities", type=int, default=10000)
    parser.add_argument("--max-triples", type=int, default=50000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        from nltk.corpus import wordnet

        wordnet.ensure_loaded()
    except (ImportError, LookupError):
        parser.error("WordNet corpus missing; run python -m nltk.downloader wordnet")
    original = json.loads(args.mapping.read_text(encoding="utf-8"))
    ranking = json.loads(args.ranking.read_text(encoding="utf-8"))
    mapping_files, graph_files, metadata = generate_outputs(
        original,
        ranking,
        args.review_csv.read_bytes(),
        wordnet=wordnet,
        max_hops=args.max_hops,
        max_entities=args.max_entities,
        max_triples=args.max_triples,
    )
    all_outputs = {args.output_root / "mapping": mapping_files, args.output_root / "kg": graph_files}
    _ensure_safe_outputs(args.output_root, all_outputs)
    for output_dir, files in all_outputs.items():
        _save_outputs(output_dir, files, dry_run=args.dry_run)
    print(json.dumps(metadata, indent=2))
    print("[AI DRAFT ONLY] No human review performed; never use these KG artifacts as final training inputs.")


if __name__ == "__main__":
    main()
