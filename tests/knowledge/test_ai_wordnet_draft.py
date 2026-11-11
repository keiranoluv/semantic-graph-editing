"""AI-provisional gate and reproducible KG export, without external corpora."""

import copy
import csv
import io
import json

import pytest

from scripts.build_ai_wordnet_draft import (
    AI_REVIEWER,
    DRAFT_LABEL,
    _ensure_safe_outputs,
    generate_outputs,
    prepare_draft,
)
from src.knowledge.sense_review import _sha256_json


class FakeSynset:
    def __init__(self, name):
        self._name = name

    def name(self):
        return self._name

    def pos(self):
        return "n"

    def hypernyms(self):
        return []

    def instance_hypernyms(self):
        return []

    def part_meronyms(self):
        return []

    def member_meronyms(self):
        return []

    def substance_meronyms(self):
        return []


class FakeWN:
    def get_version(self):
        return "3.0"

    def synset(self, name):
        return FakeSynset(name)


def fixtures():
    rows = [{"class_id": 0, "label": "__background__", "status": "background", "synset_id": None}]
    rank_rows = []
    review_rows = []
    for i in range(1, 151):
        status = "linked_auto" if i <= 23 else "ambiguous"
        sense = f"class_{i}.n.01"
        label = f"class {i}"
        rows.append(
            {
                "class_id": i,
                "label": label,
                "status": status,
                "synset_id": sense if status == "linked_auto" else None,
                "candidates": [{"synset_id": sense, "definition": f"synthetic class {i}"}],
                "match_rule": "exact_lemma",
            }
        )
        rank_rows.append(
            {
                "class_id": i,
                "label": label,
                "status": status,
                "suggested_synset": sense,
                "ranked_candidates": [{"synset_id": sense, "definition": f"synthetic class {i}"}],
            }
        )
        accepting = i <= 124
        review_rows.append(
            {
                "class_id": i,
                "label": label,
                "status": status,
                "suggested_synset": sense,
                "selected_synset": sense if accepting else "",
                "decision": "accept" if accepting else "defer",
                "reviewer": AI_REVIEWER if accepting else "",
                "notes": "AI provisional; needs human checking" if accepting else "Needs images",
            }
        )
    mapping = {"schema_version": 1, "source": "wordnet", "source_version": "3.0", "rows": rows}
    ranking = {
        "schema_version": 1,
        "source_version": "3.0",
        "mapping_sha256": _sha256_json(mapping),
        "rows": rank_rows,
    }
    return mapping, ranking, review_rows


def to_csv(rows):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode()


def test_124_provisional_and_26_deferred():
    original, rank, records = fixtures()
    draft, report, queue = prepare_draft(original, rank, to_csv(records))
    assert original["rows"][24]["status"] == "ambiguous"
    assert draft["rows"][24]["status"] == "linked_manual"  # Compatibility only.
    assert draft["rows"][24]["verified_by_human"] is False
    assert draft["rows"][125]["synset_id"] is None
    assert draft["overrides_sha256"] is None
    assert report["original_auto_count"] == 23
    assert report["new_ai_provisional_count"] == 101
    assert report["ai_provisional_class_count"] == 124
    assert report["deferred_class_count"] == 26
    assert report["human_verified_class_count"] == 0
    assert len(list(csv.DictReader(io.StringIO(queue.decode())))) == 26
    assert report["curation_tier"] == DRAFT_LABEL


def test_graph_never_reports_training_ready():
    mapping, rank, reviews = fixtures()
    files, kg, metadata = generate_outputs(mapping, rank, to_csv(reviews), wordnet=FakeWN())
    assert metadata["root_count"] == 124
    assert metadata["resolved_class_count"] == 124
    assert metadata["unreviewed_class_ids"] == list(range(1, 125))
    assert metadata["unresolved_class_ids"] == list(range(125, 151))
    assert metadata["ready_for_gat_kg_training"] is False
    assert json.loads(kg["metadata.json"])["curation_tier"] == DRAFT_LABEL
    assert len(json.loads(kg["class_entity_ids.json"])["class_entity_ids"]) == 151
    assert files["wordnet_mapping.json"].startswith(b"{\n")
    assert (files, kg, metadata) == generate_outputs(mapping, rank, to_csv(reviews), wordnet=FakeWN())


def test_rejects_mapping_hash_mismatch():
    mapping, rank, reviews = fixtures()
    rank["mapping_sha256"] = "bad"
    with pytest.raises(ValueError, match="SHA256"):
        prepare_draft(mapping, rank, to_csv(reviews))


def test_rejects_bad_review_identity_or_unsupported_synset():
    mapping, rank, reviews = fixtures()
    bad = copy.deepcopy(reviews)
    bad[23]["reviewer"] = "Human reviewer"
    with pytest.raises(ValueError, match="AI source declaration"):
        prepare_draft(mapping, rank, to_csv(bad))
    bad = copy.deepcopy(reviews)
    bad[23]["selected_synset"] = "other.n.01"
    with pytest.raises(ValueError, match="not in original"):
        prepare_draft(mapping, rank, to_csv(bad))
    bad = copy.deepcopy(reviews)
    bad[0]["selected_synset"] = "other.n.01"
    with pytest.raises(ValueError, match="may not silently change|not in original"):
        prepare_draft(mapping, rank, to_csv(bad))


def test_rejects_misalignment_and_draft_with_fake_human_review():
    mapping, rank, reviews = fixtures()
    bad = copy.deepcopy(reviews)
    bad[40]["class_id"] = 999
    with pytest.raises(ValueError, match="alignment"):
        prepare_draft(mapping, rank, to_csv(bad))
    bad = copy.deepcopy(reviews)
    bad[130]["selected_synset"] = "class_131.n.01"
    with pytest.raises(ValueError, match="Deferred row"):
        prepare_draft(mapping, rank, to_csv(bad))


def test_reject_different_existing_outputs_without_changes(tmp_path):
    output = tmp_path / "root" / "mapping"
    output.mkdir(parents=True)
    (output / "test.json").write_bytes(b"original")
    with pytest.raises(FileExistsError, match="Refusing"):
        _ensure_safe_outputs(tmp_path / "root", {output: {"test.json": b"changed"}})
    assert (output / "test.json").read_bytes() == b"original"
    _ensure_safe_outputs(tmp_path / "root", {output: {"test.json": b"original"}})
