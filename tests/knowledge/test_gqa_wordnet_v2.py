"""Explicit sample decisions cannot approve unseen classes or change frozen IDs."""

import copy

import pytest

from tools.prepare_gqa_wordnet_v2 import apply_review, audit_objects


def fixture():
    rows = [{"class_id": 0, "label": "__background__", "status": "background", "synset_id": None, "candidates": []}]
    for i, label in enumerate(("new", "changed", "missing", "unreviewed"), 1):
        rows.append(
            {
                "class_id": i,
                "label": label,
                "status": "ambiguous" if i == 1 else "unmatched" if i == 3 else "linked_auto",
                "synset_id": None if i in (1, 3) else f"{label}.n.01",
                "candidates": [] if i == 3 else [{"synset_id": f"{label}.n.01"}, {"synset_id": f"{label}.n.02"}],
            }
        )
    base = {"schema_version": 1, "source": "wordnet", "rows": rows}
    choices = [
        {
            "class_id": "1",
            "label": "new",
            "decision": "accept",
            "selected_synset": "new.n.01",
            "reviewer": "Human fixture",
            "notes": "explicit choice",
        },
        {
            "class_id": "2",
            "label": "changed",
            "decision": "accept",
            "selected_synset": "changed.n.02",
            "reviewer": "Human fixture",
            "notes": "",
        },
        {
            "class_id": "3",
            "label": "missing",
            "decision": "defer",
            "selected_synset": "",
            "reviewer": "Human fixture",
            "notes": "no candidate",
        },
    ]
    return base, choices


def test_review_changes_only_explicit_classes_and_preserves_all_unreviewed_rows():
    base, choices = fixture()
    original = copy.deepcopy(base)
    mapping, changes = apply_review(base, choices, provenance={"csv_sha256": "synthetic"})
    assert base == original
    assert mapping["rows"][4] == original["rows"][4]
    assert mapping["rows"][1]["synset_id"] == "new.n.01"
    assert mapping["rows"][2]["synset_id"] == "changed.n.02"
    assert mapping["rows"][3]["synset_id"] is None
    assert len(changes) == 2
    assert mapping["gqa_review"]["accept_count"] == 2 and mapping["gqa_review"]["defer_count"] == 1
    assert mapping["gqa_review"]["all_classes_human_reviewed"] is False


@pytest.mark.parametrize("mutation", ["duplicate", "blank", "candidate", "name", "id", "defer_sense"])
def test_invalid_or_implicit_decisions_fail_closed(mutation):
    base, choices = fixture()
    if mutation == "duplicate":
        choices.append(choices[0])
    elif mutation == "blank":
        choices[0]["decision"] = ""
    elif mutation == "candidate":
        choices[0]["selected_synset"] = "invented.n.01"
    elif mutation == "name":
        choices[0]["reviewer"] = ""
    elif mutation == "id":
        choices[0]["class_id"] = "4"
    else:
        choices[2]["selected_synset"] = "invented.n.01"
    with pytest.raises(ValueError):
        apply_review(base, choices, provenance={})


def test_full_object_audit_uses_instance_counts_and_image_counts_separately():
    base, choices = fixture()
    mapping, _ = apply_review(base, choices, provenance={})
    records = [
        {"objects": [{"label_id": i} for i in [1, 1, 2, 3]]},
        {"objects": [{"label_id": i} for i in [3, 4]]},
        {"objects": []},
    ]
    result = audit_objects(records, mapping, base)
    assert result["train_images"] == 3 and result["train_objects"] == 6
    assert result["linked_objects"] == 4 and result["v1_linked_objects"] == 2
    assert result["object_coverage"] == 4 / 6
    assert result["per_class"][0]["train_objects"] == 2 and result["per_class"][0]["train_images"] == 1
    assert [r["label"] for r in result["unlinked_by_train_frequency"]] == ["missing"]
    with pytest.raises(ValueError, match="outside"):
        audit_objects([{"objects": [{"label_id": 9}]}], mapping, base)
