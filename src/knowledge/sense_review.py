"""Reproducible, human-reviewed WordNet sense disambiguation for VG150.

A ranking is a *suggestion*, not a verified link. Only an explicit accept
choice in a review CSV is compiled to a curated WordNet override.
"""

import csv
import hashlib
import io
import json
import math
from typing import Any

REVIEW_FIELDS = (
    "class_id",
    "label",
    "status",
    "suggested_synset",
    "selected_synset",
    "decision",
    "reviewer",
    "notes",
)


def _sha256_json(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _validate_mapping(mapping: dict) -> list[dict]:
    """Reject malformed or reordered class IDs before any review work."""
    rows = mapping.get("rows")
    if mapping.get("schema_version") != 1 or mapping.get("source") != "wordnet" or not isinstance(rows, list):
        raise ValueError("Expected Milestone A WordNet mapping with schema_version 1")
    if not rows or rows[0].get("class_id") != 0 or rows[0].get("status") != "background":
        raise ValueError("Mapping must start with background class ID 0")
    for idx, row in enumerate(rows):
        if type(row.get("class_id")) is not int or row["class_id"] != idx:
            raise ValueError("Mapping class IDs must be contiguous and ordered")
        if idx and (
            not isinstance(row.get("label"), str)
            or row.get("status") not in {"linked_auto", "linked_manual", "ambiguous", "unmatched"}
            or not isinstance(row.get("candidates"), list)
        ):
            raise ValueError(f"Invalid mapping entry at ID {idx}")
    return rows


def rank_mapping(mapping: dict, *, wordnet: Any) -> dict:
    """Rank lexical candidates deterministically; do not auto-accept them.

    A deliberately simple, auditable heuristic uses exact lemma identity,
    WordNet lemma usage counts, and a tiny canonical-sense tiebreaker.
    It is NOT semantic correctness confidence or a calibrated probability.
    """
    rows = _validate_mapping(mapping)
    if str(wordnet.get_version()) != str(mapping.get("source_version")):
        raise ValueError("WordNet version mismatch between mapping and ranking corpus")
    output = []
    for row in rows[1:]:
        label = row["label"]
        normalized = row.get("normalized_label") or label.strip().casefold().replace(" ", "_").replace("-", "_")
        ranked = []
        for candidate in row["candidates"]:
            name = candidate["synset_id"]
            synset = wordnet.synset(name)
            if synset.pos() != "n" or synset.name() != name:
                raise ValueError(f"Invalid candidate noun synset: {name}")
            lemmas = list(synset.lemmas())
            exact = any(lemma.name().casefold() == normalized for lemma in lemmas)
            matched_counts = [max(0, int(lemma.count())) for lemma in lemmas if lemma.name().casefold() == normalized]
            usage_count = max(matched_counts, default=0)
            # No VG data, labels, relation counts, images or external LLM calls.
            try:
                sense_number = int(name.rsplit(".", 1)[-1])
            except ValueError:
                sense_number = 1000
            score = round(100.0 * exact + 8.0 * math.log1p(usage_count) + 0.1 / max(1, sense_number), 6)
            ranked.append(
                {
                    "synset_id": name,
                    "definition": candidate.get("definition", ""),
                    "score": score,
                    "exact_lemma": exact,
                    "lemma_usage_count": usage_count,
                }
            )
        ranked.sort(key=lambda item: (-item["score"], item["synset_id"]))
        output.append(
            {
                "class_id": row["class_id"],
                "label": label,
                "status": row["status"],
                "suggested_synset": ranked[0]["synset_id"] if ranked else "",
                "ranked_candidates": ranked,
            }
        )
    return {
        "schema_version": 1,
        "source_version": mapping["source_version"],
        "mapping_sha256": _sha256_json(mapping),
        "ranking_policy": "exact_lemma_100+8log1p(lemma_count)+0.1/sense_number",
        "warning": "Ranking is a lexical heuristic; every accept requires human review.",
        "rows": output,
    }


def review_csv(ranking: dict) -> bytes:
    """Generate editable review sheet for ALL object classes, including auto links."""
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=REVIEW_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in ranking["rows"]:
        writer.writerow(
            {
                "class_id": row["class_id"],
                "label": row["label"],
                "status": row["status"],
                "suggested_synset": row["suggested_synset"],
                "selected_synset": "",
                "decision": "",
                "reviewer": "",
                "notes": "",
            }
        )
    return stream.getvalue().encode("utf-8")


def compile_reviews(mapping: dict, review_csv_bytes: bytes, base_overrides: dict) -> tuple[dict, dict]:
    """Accept only explicit reviewed decisions grounded in original candidates."""
    rows = _validate_mapping(mapping)[1:]
    if base_overrides.get("version") != 1 or not isinstance(base_overrides.get("mappings"), dict):
        raise ValueError("Expected existing overrides in version 1 format")
    original = base_overrides["mappings"]
    names = {row["label"] for row in rows}
    if any(key not in names for key in original):
        raise ValueError("Base overrides contain unknown VG150 labels")
    reader = csv.DictReader(io.StringIO(review_csv_bytes.decode("utf-8-sig"), newline=""))
    if not set(REVIEW_FIELDS).issubset(reader.fieldnames or []):
        raise ValueError("Review CSV columns changed")
    review_rows = list(reader)
    if len(review_rows) != len(rows):
        raise ValueError("Review CSV must retain exactly one row per VG150 object class")
    curated = dict(original)
    provenance = dict(base_overrides.get("review_provenance", {}))
    accepted = 0
    for item, expected in zip(review_rows, rows, strict=True):
        if (item["class_id"], item["label"], item["status"]) != (
            str(expected["class_id"]),
            expected["label"],
            expected["status"],
        ):
            raise ValueError(f"Review CSV class alignment mismatch at ID {expected['class_id']}")
        decision = item["decision"].strip().casefold()
        choice = item["selected_synset"].strip()
        reviewer = item["reviewer"].strip()
        notes = item["notes"].strip()
        if decision not in {"", "defer", "accept", "reject"}:
            raise ValueError(f"Unknown decision at ID {expected['class_id']}: {decision}")
        if decision != "accept":
            if choice:
                raise ValueError(f"Class {expected['class_id']}: choice without accept decision")
            continue
        allowed = {candidate["synset_id"] for candidate in expected["candidates"]}
        if not choice or choice not in allowed:
            raise ValueError(f"Class {expected['class_id']}: chosen synset not in source candidates")
        if not reviewer:
            raise ValueError(f"Class {expected['class_id']}: reviewer identity required")
        if expected["status"] in {"ambiguous", "unmatched"} and not notes:
            raise ValueError(f"Class {expected['class_id']}: rationale required for ambiguous/unmatched label")
        if expected["label"] in curated and curated[expected["label"]] != choice:
            raise ValueError(f"Class {expected['class_id']}: conflicts with existing override")
        curated[expected["label"]] = choice
        provenance[expected["label"]] = {
            "class_id": expected["class_id"],
            "reviewer": reviewer,
            "notes": notes,
            "candidate_source": "wordnet",
            "source_version": mapping["source_version"],
        }
        accepted += 1
    result = {
        "version": 1,
        "mappings": dict(sorted(curated.items())),
        "review_provenance": dict(sorted(provenance.items())),
    }
    return result, {
        "accepted_decisions": accepted,
        "total_overrides": len(curated),
        "class_count": len(rows),
        "mapping_sha256": _sha256_json(mapping),
        "review_csv_sha256": hashlib.sha256(review_csv_bytes).hexdigest(),
    }
