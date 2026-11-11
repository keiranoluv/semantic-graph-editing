"""Conservative, auditable WordNet linking for the VG150 object vocabulary.

This module reads object class names only. No images, boxes, predicate
annotations, or VG150 train/validation/test samples are accessed.
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Any

BACKGROUND_LABEL = "__background__"


def _normalize(label: str) -> str:
    """Normalize a label without inventing its meaning."""
    return re.sub(r"[\s-]+", "_", label.strip().casefold())


def _read_classes(vocabulary_path: Path, expected_classes: int) -> tuple[list[tuple[int, str]], str]:
    """Validate class-to-ID alignment (including reserved background row 0)."""
    if expected_classes <= 0:
        raise ValueError("expected_classes must be positive")
    raw = vocabulary_path.read_bytes()
    vocab = json.loads(raw)
    label_to_idx = vocab.get("label_to_idx")
    if not isinstance(label_to_idx, dict):
        raise ValueError("Vocabulary requires label_to_idx object")

    class_by_id: dict[int, str] = {}
    for label, index in label_to_idx.items():
        if not isinstance(label, str) or type(index) is not int:
            raise ValueError("label_to_idx must map strings to integer class IDs")
        if index == 0:
            if label != BACKGROUND_LABEL:
                raise ValueError("ID 0 is reserved for __background__")
            continue
        if index < 0 or index in class_by_id:
            raise ValueError(f"Invalid or repeated class ID: {index}")
        class_by_id[index] = label

    expected_ids = set(range(1, expected_classes + 1))
    if set(class_by_id) != expected_ids:
        raise ValueError("Object classes must occupy exactly IDs 1.." f"{expected_classes}; got {len(class_by_id)} IDs")

    idx_to_label = vocab.get("idx_to_label")
    if idx_to_label is not None:
        if not isinstance(idx_to_label, dict):
            raise ValueError("idx_to_label must be a JSON object")
        for index, label in class_by_id.items():
            if str(index) not in idx_to_label or idx_to_label[str(index)] != label:
                raise ValueError(f"idx_to_label mismatch at class ID {index}")

    return sorted(class_by_id.items()), hashlib.sha256(raw).hexdigest()


def _load_overrides(overrides_path: Path | None, class_names: set[str]) -> tuple[dict[str, str], str | None]:
    """Read explicit, reviewed noun synsets keyed by exact VG150 class names."""
    if overrides_path is None:
        return {}, None
    raw = overrides_path.read_bytes()
    overrides = json.loads(raw)
    if not isinstance(overrides, dict) or overrides.get("version") != 1:
        raise ValueError("Overrides require schema version 1")
    mapping = overrides.get("mappings")
    if not isinstance(mapping, dict):
        raise ValueError("Overrides require a mappings object")
    for label, synset in mapping.items():
        if label not in class_names or not isinstance(synset, str) or not synset:
            raise ValueError(f"Invalid override for class {label!r}")
    return mapping, hashlib.sha256(raw).hexdigest()


def _synset_record(synset: Any) -> dict[str, str]:
    return {"synset_id": synset.name(), "definition": synset.definition()}


def _candidates(wordnet: Any, normalized: str) -> tuple[list[Any], str]:
    """Find noun senses, trying a morphological normalization only as fallback."""
    direct = wordnet.synsets(normalized, pos="n")
    match_rule = "exact_lemma"
    if not direct:
        base = wordnet.morphy(normalized, pos="n")
        if base is not None and base != normalized:
            direct = wordnet.synsets(base, pos="n")
            match_rule = "noun_morphology"
    seen: set[str] = set()
    unique = []
    for synset in direct:
        if synset.name() not in seen:
            unique.append(synset)
            seen.add(synset.name())
    return sorted(unique, key=lambda sense: sense.name()), match_rule


def link_vocabulary(
    vocabulary_path: Path,
    *,
    wordnet: Any,
    expected_classes: int = 150,
    overrides_path: Path | None = None,
    nltk_version: str = "unspecified",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Link VG150 object class IDs conservatively to WordNet noun senses.

    Auto-resolve only labels with a single candidate noun synset. Multiple
    senses remain ambiguous unless explicitly curated; zero senses remain
    unmatched. Candidate coverage is reported separately from resolved coverage.
    """
    vocabulary_path = Path(vocabulary_path)
    classes, vocabulary_sha256 = _read_classes(vocabulary_path, expected_classes)
    overrides, overrides_sha256 = _load_overrides(overrides_path, {name for _, name in classes})

    rows: list[dict[str, Any]] = [
        {
            "class_id": 0,
            "label": BACKGROUND_LABEL,
            "normalized_label": None,
            "status": "background",
            "synset_id": None,
            "match_rule": None,
            "candidates": [],
        }
    ]

    for class_id, label in classes:
        normalized = _normalize(label)
        candidates, rule = _candidates(wordnet, normalized)
        selected = None
        status = "unmatched"
        if label in overrides:
            selected = overrides[label]
            chosen = wordnet.synset(selected)  # fails on invalid overrides
            if chosen.pos() != "n" or chosen.name() != selected:
                raise ValueError(f"Non-noun or invalid override for {label!r}: {selected!r}")
            status = "linked_manual"
            rule = "curated_override"
        elif len(candidates) == 1:
            selected = candidates[0].name()
            status = "linked_auto"
        elif len(candidates) > 1:
            status = "ambiguous"

        rows.append(
            {
                "class_id": class_id,
                "label": label,
                "normalized_label": normalized,
                "status": status,
                "synset_id": selected,
                "match_rule": rule if candidates or selected else None,
                "candidates": [_synset_record(synset) for synset in candidates],
            }
        )

    counts = {
        status: sum(r["status"] == status for r in rows[1:])
        for status in ("linked_auto", "linked_manual", "ambiguous", "unmatched")
    }
    resolved = counts["linked_auto"] + counts["linked_manual"]
    with_candidates = sum(bool(r["candidates"]) for r in rows[1:])
    report = {
        "class_count": expected_classes,
        **counts,
        "resolved_count": resolved,
        "resolved_coverage": resolved / expected_classes,
        "candidate_count": with_candidates,
        "candidate_coverage": with_candidates / expected_classes,
        "requires_review_count": counts["ambiguous"] + counts["unmatched"],
        "policy": "single_noun_sense_or_curated_override",
    }
    mapping = {
        "schema_version": 1,
        "source": "wordnet",
        "source_version": str(wordnet.get_version()),
        "nltk_version": nltk_version,
        "vocabulary_sha256": vocabulary_sha256,
        "overrides_sha256": overrides_sha256,
        "rows": rows,
    }
    return mapping, report
