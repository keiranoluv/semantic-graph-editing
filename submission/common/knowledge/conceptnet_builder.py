"""Auditable, offline ConceptNet 5.7 extraction anchored on VG150 label text.

Lexical matches are only candidates. This module never asserts WordNet-
ConceptNet identity and never reads Visual Genome relation annotations.
"""

import csv
import gzip
import hashlib
import io
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import unquote

CONCEPTNET_VERSION = "5.7.0"
RELATIONS = (
    "IsA",
    "PartOf",
    "HasA",
    "MadeOf",
    "UsedFor",
    "CapableOf",
    "AtLocation",
    "HasProperty",
    "ReceivesAction",
    "RelatedTo",
)
SUPPORTED_LICENSES = {"cc:by/4.0", "cc:by-sa/4.0", "cc:by-sa/3.0", "cc:by/3.0"}
RELATION_URIS = frozenset(f"/r/{name}" for name in RELATIONS)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _label_key(label: str) -> str:
    return re.sub(r"[\s-]+", "_", label.strip().casefold())


def _concept_key(uri: str, *, allow_pos: bool = True) -> str | None:
    """Return lexical form of a simple English ConceptNet URI; never a sense ID."""
    fields = uri.split("/")
    if len(fields) not in (4, 5) or fields[:3] != ["", "c", "en"]:
        return None
    if len(fields) == 5 and (not allow_pos or fields[4] not in ("n", "v", "a", "r")):
        return None
    term = unquote(fields[3])
    if not term or "/" in term or any(char.isspace() for char in term):
        return None
    return term.casefold()


def validate_mapping(mapping: dict) -> list[dict]:
    """Accept A.1 or AI-draft WordNet map without treating any link as reviewed."""
    rows = mapping.get("rows")
    if mapping.get("schema_version") != 1 or mapping.get("source") != "wordnet":
        raise ValueError("Expected schema_version=1 WordNet class mapping")
    if not isinstance(rows, list) or len(rows) != 151:
        raise ValueError("Expected 151 VG150 rows (background plus 150 labels)")
    if rows[0].get("class_id") != 0 or rows[0].get("status") != "background":
        raise ValueError("VG150 class ID 0 must be background")
    names = set()
    for idx, row in enumerate(rows[1:], start=1):
        if type(row.get("class_id")) is not int or row["class_id"] != idx:
            raise ValueError(f"Invalid VG150 class ordering at ID {idx}")
        label = row.get("label")
        if not isinstance(label, str) or not label.strip() or label in names:
            raise ValueError(f"Invalid or duplicate VG150 label at ID {idx}")
        names.add(label)
    return rows


def extract_assertions(
    mapping: dict,
    assertions: Path,
    *,
    expected_sha256: str,
    min_weight: float = 1.0,
    max_edges: int = 50000,
) -> tuple[dict[str, bytes], dict]:
    """Stream pinned 5-column ConceptNet assertions, preserving edge evidence.

    Filtering keeps only edges incident on an EXACT VG150 English lexeme.
    No endpoint is mapped to a WordNet synset; no external HTTP calls occur.
    """
    rows = validate_mapping(mapping)
    assertions = Path(assertions)
    if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
        raise ValueError("--expected-sha256 must contain exactly 64 hexadecimal characters")
    if not (math.isfinite(min_weight) and min_weight > 0) or max_edges <= 0:
        raise ValueError("Invalid weight or edge limit")
    actual_sha256 = _sha256_file(assertions)
    if actual_sha256 != expected_sha256.lower():
        raise ValueError("ConceptNet snapshot SHA256 mismatch; refusing to extract")

    keys = defaultdict(list)
    for row in rows[1:]:
        keys[_label_key(row["label"])].append(row["class_id"])
    counts = Counter()
    # Retain evidence with its assertion URI, original dataset and license.
    evidence = set()
    opener = gzip.open if assertions.name.endswith(".gz") else open
    with opener(assertions, "rt", encoding="utf-8", newline="") as stream:
        for line in stream:
            counts["input_rows"] += 1
            parts = line.rstrip("\r\n").split("\t", 4)
            if len(parts) != 5:
                raise ValueError(f"Malformed ConceptNet 5-field assertion at line {counts['input_rows']}")
            assertion_uri, rel, start, end, metadata_json = parts
            if rel not in RELATION_URIS:
                counts["filtered_relation"] += 1
                continue
            skey = _concept_key(start)
            ekey = _concept_key(end)
            if skey is None or ekey is None:
                counts["filtered_non_english_or_complex_uri"] += 1
                continue
            start_match = skey in keys and (start.count("/") == 3 or start.endswith("/n"))
            end_match = ekey in keys and (end.count("/") == 3 or end.endswith("/n"))
            if not (start_match or end_match):
                counts["filtered_not_incident_on_class_label"] += 1
                continue
            try:
                metadata = json.loads(metadata_json)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed assertion metadata at line {counts['input_rows']}") from exc
            if not isinstance(metadata, dict):
                raise ValueError(f"Metadata must be an object at line {counts['input_rows']}")
            dataset = metadata.get("dataset")
            license_id = metadata.get("license")
            weight = metadata.get("weight")
            if not isinstance(dataset, str) or not isinstance(license_id, str):
                raise ValueError(f"Missing dataset/license at line {counts['input_rows']}")
            if dataset.startswith("/d/wordnet"):
                counts["filtered_wordnet_derived"] += 1
                continue
            if license_id not in SUPPORTED_LICENSES:
                counts["filtered_unrecognized_license"] += 1
                continue
            if type(weight) not in (int, float) or not math.isfinite(weight) or weight <= 0:
                raise ValueError(f"Invalid assertion weight at line {counts['input_rows']}")
            if weight < min_weight:
                counts["filtered_below_min_weight"] += 1
                continue
            if not assertion_uri.startswith("/a/"):
                raise ValueError(f"Invalid assertion URI at line {counts['input_rows']}")
            evidence.add((start, rel[3:], end, assertion_uri, dataset, license_id, float(weight)))
            if len(evidence) > max_edges:
                raise ValueError("B2 edge limit exceeded; refine the filters or set an explicit larger --max-edges")
    if not evidence:
        raise ValueError("No ConceptNet assertions matched VG150 class labels and filters")
    entities = sorted({f"cn:{node}" for start, _, end, *_ in evidence for node in (start, end)})
    eid = {name: idx for idx, name in enumerate(entities)}
    rels = [rel for rel in RELATIONS if any(row[1] == rel for row in evidence)]
    rid = {name: idx for idx, name in enumerate(rels)}

    lines = io.StringIO(newline="")
    writer = csv.writer(lines, delimiter="\t", lineterminator="\n")
    writer.writerow(("subject_id", "relation_id", "object_id", "assertion_uri", "dataset", "license", "weight"))
    unique_triples = set()
    class_to_candidates = defaultdict(set)
    per_class_edges = Counter()
    for start, rel, end, uri, dataset, license_id, weight in sorted(evidence):
        writer.writerow((eid[f"cn:{start}"], rid[rel], eid[f"cn:{end}"], uri, dataset, license_id, weight))
        unique_triples.add((start, rel, end))
        for node in (start, end):
            key = _concept_key(node)
            if key in keys and (node.count("/") == 3 or node.endswith("/n")):
                for class_id in keys[key]:
                    class_to_candidates[class_id].add(node)
                    per_class_edges[class_id] += 1

    links = [{"class_id": 0, "label": "__background__", "concept_ids": [], "human_verified": False}]
    review = io.StringIO(newline="")
    review_writer = csv.writer(review, lineterminator="\n")
    review_writer.writerow(
        (
            "class_id",
            "label",
            "wordnet_status",
            "wordnet_synset",
            "conceptnet_uri_candidates",
            "edge_evidence_count",
            "lexical_decision",
            "sense_alignment_decision",
            "reviewer",
            "notes",
        )
    )
    for row in rows[1:]:
        idx = row["class_id"]
        candidates = sorted(class_to_candidates.get(idx, set()))
        links.append(
            {
                "class_id": idx,
                "label": row["label"],
                "concept_ids": [eid[f"cn:{c}"] for c in candidates],
                "concept_uris": candidates,
                "match_type": "UNVERIFIED_EXACT_LEXICAL_MATCH" if candidates else "NO_MATCH",
                "human_verified": False,
            }
        )
        review_writer.writerow(
            (
                idx,
                row["label"],
                row.get("status", ""),
                row.get("synset_id") or "",
                ";".join(candidates),
                per_class_edges[idx],
                "",
                "",
                "",
                "",
            )
        )
    metadata = {
        "schema_version": 1,
        "source": "conceptnet",
        "source_version": CONCEPTNET_VERSION,
        "source_snapshot_sha256": actual_sha256,
        "input_mapping_canonical_sha256": hashlib.sha256(
            (json.dumps(mapping, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        ).hexdigest(),
        "class_count_including_background": 151,
        "class_count_with_lexical_candidates": sum(bool(class_to_candidates.get(i)) for i in range(1, 151)),
        "class_count_without_lexical_candidates": sum(not class_to_candidates.get(i) for i in range(1, 151)),
        "entity_count": len(entities),
        "relation_count": len(rels),
        "edge_evidence_count": len(evidence),
        "unique_triple_count": len(unique_triples),
        "max_edges": max_edges,
        "min_weight": min_weight,
        "allowed_relations": list(RELATIONS),
        "excluded_wordnet_derived_assertions": True,
        "filter_counts": dict(sorted(counts.items())),
        "license_policy": "ConceptNet 5 data CC BY-SA 4.0 overall; preserve each assertion license and attribution",
        "curation_tier": "B2_LEXICAL_CANDIDATES_NOT_HUMAN_VERIFIED",
        "wordnet_conceptnet_sameas_links": 0,
        "human_verified_class_count": 0,
        "ready_for_gat_kg_training": False,
        "leakage_policy": "vocabulary_labels_and_external_conceptnet_only_no_vg_predicates_or_images",
    }

    def to_json(obj):
        return (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8")

    return {
        "entities.json": to_json({"entities": entities}),
        "relations.json": to_json({"relations": rels}),
        "triples.tsv": lines.getvalue().encode("utf-8"),
        "class_concept_candidates.json": to_json({"classes": links}),
        "review_queue.csv": review.getvalue().encode("utf-8"),
        "metadata.json": to_json(metadata),
    }, metadata
