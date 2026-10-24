"""The review worksheet must expose deferred AI hints without accepting them."""

import copy

import pytest

from scripts.prepare_level1_review import prepare_rows
from src.knowledge.sense_review import _sha256_json


def inputs():
    rows = [{"class_id": 0, "label": "__background__", "status": "background", "synset_id": None}]
    draft_rows = copy.deepcopy(rows)
    for i in range(1, 151):
        candidate = {"synset_id": "object.n.01", "definition": "an object"}
        rows.append(
            {
                "class_id": i,
                "label": f"class{i}",
                "status": "linked_auto" if i != 150 else "ambiguous",
                "candidates": [candidate],
            }
        )
        draft_rows.append(
            {
                "class_id": i,
                "label": f"class{i}",
                "status": "linked_auto" if i != 150 else "ambiguous",
                "synset_id": "object.n.01" if i != 150 else None,
                "candidates": [candidate],
            }
        )
    original = {"schema_version": 1, "source": "wordnet", "rows": rows}
    draft = {
        "schema_version": 1,
        "source": "wordnet",
        "rows": draft_rows,
        "source_mapping_canonical_sha256": _sha256_json(original),
    }
    deferred = [
        {"class_id": "150", "label": "class150", "suggested_synset": "object.n.01", "reason": "needs image context"}
    ]
    return original, draft, deferred


def test_deferred_suggestion_is_visible_but_not_accepted():
    rows = prepare_rows(*inputs())
    assert len(rows) == 150
    assert all(row["suggested_synset"] == "object.n.01" for row in rows)
    assert all(row["selected_synset"] == row["decision"] == row["reviewer"] == "" for row in rows)
    assert rows[-1]["draft_mapping_state"] == "deferred_unmapped"
    assert rows[-1]["deferred_reason"] == "needs image context"


def test_missing_or_misaligned_deferred_hint_fails():
    original, draft, deferred = inputs()
    with pytest.raises(ValueError, match="disagree"):
        prepare_rows(original, draft, [])
    deferred[0]["suggested_synset"] = "wrong.n.01"
    with pytest.raises(ValueError, match="outside original"):
        prepare_rows(original, draft, deferred)
