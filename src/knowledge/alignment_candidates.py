"""B.2.2 candidate-only VG150 WordNet / ConceptNet alignment.

This module constructs an *audit queue*, not semantic identity edges. No
Visual Genome image/annotation or predicate data is accessed.
"""

import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
CURATION_TIER = "CANDIDATE_BRIDGE_NOT_HUMAN_VERIFIED"
PAIR_FIELDS = (
    "class_id",
    "label",
    "wordnet_synset",
    "wordnet_entity_id",
    "conceptnet_uri",
    "conceptnet_entity_id",
    "candidate_basis",
    "human_verified",
)
BRIDGE_REVIEW_FIELDS = (
    "class_id",
    "label",
    "wordnet_synset",
    "conceptnet_uri",
    "bridge_decision",
    "human_reviewer",
    "evidence_reference",
    "notes",
)
REVIEW_FIELDS = (
    "class_id",
    "label",
    "wordnet_synset",
    "wordnet_status",
    "wordnet_curation_tier",
    "conceptnet_uri_candidates",
    "wordnet_review_decision",
    "conceptnet_review_decision",
    "bridge_review_decision",
    "human_reviewer",
    "evidence_reference",
    "notes",
)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _canonical_sha(value: Any) -> str:
    raw = (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_json(path: Path) -> tuple[Any, str]:
    raw = path.read_bytes()
    return json.loads(raw), _sha_bytes(raw)


def _expect_entities(document: dict, prefix: str) -> list[str]:
    entities = document.get("entities")
    _require(isinstance(entities, list), f"{prefix} entities must be a list")
    _require(all(isinstance(x, str) and x.startswith(prefix) for x in entities), f"Invalid {prefix} entity namespace")
    _require(len(set(entities)) == len(entities), f"Duplicate {prefix} entities")
    return entities


def build_candidates(
    mapping: dict,
    wn_entities_doc: dict,
    wn_class_entity_doc: dict,
    wn_metadata: dict,
    cn_entities_doc: dict,
    cn_class_candidates_doc: dict,
    cn_metadata: dict,
    *,
    mapping_bytes_sha256: str,
    input_sha256: dict[str, str],
) -> tuple[dict[str, bytes], dict]:
    """Build auditable pair candidates, with NO identity or approved bridge edges."""
    rows = mapping.get("rows")
    _require(mapping.get("schema_version") == 1 and mapping.get("source") == "wordnet", "Expected WordNet mapping v1")
    _require(isinstance(rows, list) and len(rows) == 151, "WordNet mapping needs 151 rows")
    wn_entities = _expect_entities(wn_entities_doc, "wn:")
    cn_entities = _expect_entities(cn_entities_doc, "cn:")
    class_ids = wn_class_entity_doc.get("class_entity_ids")
    classes = cn_class_candidates_doc.get("classes")
    _require(isinstance(class_ids, list) and len(class_ids) == 151, "WordNet class IDs must contain 151 entries")
    _require(isinstance(classes, list) and len(classes) == 151, "ConceptNet class candidates need 151 rows")
    _require(
        wn_metadata.get("schema_version") == 1 and wn_metadata.get("source") == "wordnet",
        "Invalid WordNet KG metadata",
    )
    _require(
        wn_metadata.get("input_mapping_sha256") == mapping_bytes_sha256,
        "WordNet KG input mapping SHA256 mismatch",
    )
    _require(wn_metadata.get("source_version") == mapping.get("source_version"), "WordNet version mismatch")
    _require(
        cn_metadata.get("schema_version") == 1 and cn_metadata.get("source") == "conceptnet",
        "Invalid ConceptNet KG metadata",
    )
    _require(cn_metadata.get("source_version") == "5.7.0", "Expected ConceptNet 5.7.0")
    _require(
        cn_metadata.get("input_mapping_canonical_sha256") == _canonical_sha(mapping),
        "ConceptNet input mapping SHA256 mismatch",
    )
    _require(
        wn_metadata.get("ready_for_gat_kg_training") is False,
        "B2.2 draft requires non-training-ready WordNet KG",
    )
    _require(
        cn_metadata.get("ready_for_gat_kg_training") is False,
        "B2.2 draft requires non-training-ready ConceptNet KG",
    )
    _require(cn_metadata.get("human_verified_class_count") == 0, "B2.2 draft expects unverified ConceptNet candidates")
    _require(
        mapping.get("curation_tier") == "AI_PROVISIONAL_NOT_HUMAN_VERIFIED",
        "B2.2 must consume explicit AI-provisional mapping",
    )

    wn_linked = 0
    cn_linked = 0
    both = 0
    pair_count = 0
    missing_wn = []
    missing_cn = []
    summaries = []
    pairs_stream = io.StringIO(newline="")
    pair_writer = csv.DictWriter(pairs_stream, fieldnames=PAIR_FIELDS, delimiter="\t", lineterminator="\n")
    pair_writer.writeheader()
    bridge_review_stream = io.StringIO(newline="")
    bridge_review_writer = csv.DictWriter(bridge_review_stream, fieldnames=BRIDGE_REVIEW_FIELDS, lineterminator="\n")
    bridge_review_writer.writeheader()
    review_stream = io.StringIO(newline="")
    review_writer = csv.DictWriter(review_stream, fieldnames=REVIEW_FIELDS, lineterminator="\n")
    review_writer.writeheader()

    for i, (row, cid, cn_class) in enumerate(zip(rows, class_ids, classes, strict=True)):
        _require(type(row.get("class_id")) is int and row["class_id"] == i, f"WordNet class ordering mismatch at {i}")
        _require(
            type(cn_class.get("class_id")) is int and cn_class["class_id"] == i,
            f"ConceptNet class ordering mismatch at {i}",
        )
        _require(row.get("label") == cn_class.get("label"), f"Class label mismatch at {i}")
        uri_list = cn_class.get("concept_uris", [])
        id_list = cn_class.get("concept_ids", [])
        _require(isinstance(uri_list, list) and isinstance(id_list, list), f"Missing ConceptNet concept lists at {i}")
        _require(len(uri_list) == len(id_list), f"ConceptNet candidates length mismatch at {i}")
        _require(not cn_class.get("human_verified", False), f"Unexpected human-verified ConceptNet class {i}")
        _require(len(set(uri_list)) == len(uri_list), f"Duplicate ConceptNet candidate at {i}")
        synset = row.get("synset_id")
        wn_entity = None
        if i == 0:
            _require(
                row.get("status") == "background" and synset is None and cid is None,
                "Background WordNet ID must be null",
            )
            _require(not uri_list and not id_list, "Background ConceptNet candidates must be empty")
        elif synset is None:
            _require(
                cid is None and row.get("status") in ("ambiguous", "unmatched"),
                f"Unresolved WordNet ID must be null at {i}",
            )
            missing_wn.append(i)
        else:
            _require(isinstance(synset, str) and synset, f"Invalid synset at {i}")
            _require(type(cid) is int and 0 <= cid < len(wn_entities), f"WordNet entity ID out of range at {i}")
            _require(wn_entities[cid] == f"wn:{synset}", f"WordNet class alignment mismatch at {i}")
            _require(row.get("status") in ("linked_auto", "linked_manual"), f"WordNet link status mismatch at {i}")
            wn_linked += 1
            wn_entity = wn_entities[cid]
        for uri, concept_id in zip(uri_list, id_list, strict=True):
            _require(isinstance(uri, str) and uri.startswith("/c/en/"), f"Invalid ConceptNet candidate URI at {i}")
            _require(
                type(concept_id) is int and 0 <= concept_id < len(cn_entities),
                f"ConceptNet entity ID out of range at {i}",
            )
            _require(cn_entities[concept_id] == f"cn:{uri}", f"ConceptNet class alignment mismatch at {i}")
        if i > 0:
            if uri_list:
                cn_linked += 1
            else:
                missing_cn.append(i)
            if wn_entity and uri_list:
                both += 1
                for uri, concept_id in zip(uri_list, id_list, strict=True):
                    pair_writer.writerow(
                        {
                            "class_id": i,
                            "label": row["label"],
                            "wordnet_synset": synset,
                            "wordnet_entity_id": cid,
                            "conceptnet_uri": uri,
                            "conceptnet_entity_id": concept_id,
                            "candidate_basis": "SHARED_VG150_LABEL_ONLY_NOT_IDENTITY",
                            "human_verified": "false",
                        }
                    )
                    bridge_review_writer.writerow(
                        {
                            "class_id": i,
                            "label": row["label"],
                            "wordnet_synset": synset,
                            "conceptnet_uri": uri,
                            "bridge_decision": "",
                            "human_reviewer": "",
                            "evidence_reference": "",
                            "notes": "",
                        }
                    )
                    pair_count += 1
            review_writer.writerow(
                {
                    "class_id": i,
                    "label": row["label"],
                    "wordnet_synset": synset or "",
                    "wordnet_status": row.get("status", ""),
                    "wordnet_curation_tier": row.get("curation_tier", "UNREVIEWED"),
                    "conceptnet_uri_candidates": ";".join(uri_list),
                    "wordnet_review_decision": "",
                    "conceptnet_review_decision": "",
                    "bridge_review_decision": "",
                    "human_reviewer": "",
                    "evidence_reference": "",
                    "notes": "",
                }
            )
        summaries.append(
            {
                "class_id": i,
                "label": row["label"],
                "wordnet_synset": synset,
                "wordnet_entity_id": cid,
                "wordnet_status": row.get("status"),
                "wordnet_verified_by_human": False,
                "conceptnet_candidates": [
                    {"uri": uri, "entity_id": concept_id, "human_verified": False}
                    for uri, concept_id in zip(uri_list, id_list, strict=True)
                ],
                "bridge_status": "UNVERIFIED_CANDIDATE_ONLY",
            }
        )

    _require(wn_metadata.get("resolved_class_count") == wn_linked, "WordNet KG resolved class count mismatch")
    _require(
        cn_metadata.get("class_count_with_lexical_candidates") == cn_linked,
        "ConceptNet KG candidate count mismatch",
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "source": "wordnet_conceptnet_candidate_alignment",
        "milestone": "B.2.2",
        "curation_tier": CURATION_TIER,
        "class_count_including_background": 151,
        "class_count": 150,
        "wordnet_linked_class_count": wn_linked,
        "conceptnet_candidate_class_count": cn_linked,
        "both_candidate_class_count": both,
        "candidate_pair_count": pair_count,
        "wordnet_missing_class_ids": missing_wn,
        "conceptnet_missing_class_ids": missing_cn,
        "review_queue_class_count": 150,
        "bridge_review_queue_pair_count": pair_count,
        "human_verified_bridge_count": 0,
        "sameas_edge_count": 0,
        "training_graph_edge_count": 0,
        "ready_for_gat_kg_training": False,
        "leakage_policy": "VG150_object_names_and_external_KGs_only_no_predicate_GT",
        "wordnet_version": mapping["source_version"],
        "conceptnet_version": cn_metadata["source_version"],
        "conceptnet_snapshot_sha256": cn_metadata.get("source_snapshot_sha256"),
        "input_file_sha256": dict(sorted(input_sha256.items())),
    }
    return {
        "class_alignment_candidates.json": _json_bytes({"classes": summaries}),
        "candidate_pairs.tsv": pairs_stream.getvalue().encode("utf-8"),
        "review_queue.csv": review_stream.getvalue().encode("utf-8"),
        "bridge_review_queue.csv": bridge_review_stream.getvalue().encode("utf-8"),
        "metadata.json": _json_bytes(metadata),
    }, metadata


def load_and_build(
    *,
    wordnet_mapping: Path,
    wordnet_kg: Path,
    conceptnet_kg: Path,
) -> tuple[dict[str, bytes], dict]:
    """Load pinned B.1 / B.2.1 outputs; validate matching provenances."""
    input_paths = {
        "wordnet_mapping": wordnet_mapping,
        "wordnet_entities": wordnet_kg / "entities.json",
        "wordnet_class_entity_ids": wordnet_kg / "class_entity_ids.json",
        "wordnet_metadata": wordnet_kg / "metadata.json",
        "conceptnet_entities": conceptnet_kg / "entities.json",
        "conceptnet_class_candidates": conceptnet_kg / "class_concept_candidates.json",
        "conceptnet_metadata": conceptnet_kg / "metadata.json",
    }
    loaded = {}
    digests = {}
    for name, path in input_paths.items():
        loaded[name], digests[name] = _load_json(path)
    return build_candidates(
        loaded["wordnet_mapping"],
        loaded["wordnet_entities"],
        loaded["wordnet_class_entity_ids"],
        loaded["wordnet_metadata"],
        loaded["conceptnet_entities"],
        loaded["conceptnet_class_candidates"],
        loaded["conceptnet_metadata"],
        mapping_bytes_sha256=digests["wordnet_mapping"],
        input_sha256=digests,
    )
