"""Tests for deterministic reports and overwrite safeguards."""

import json

import pytest

from scripts.prepare_knowledge import _save_outputs, _serialize


def test_serialization_review_only_ambiguity():
    mapping = {
        "rows": [
            {"class_id": 0, "status": "background"},
            {
                "class_id": 1,
                "label": "bat",
                "status": "ambiguous",
                "synset_id": None,
                "candidates": [{"synset_id": "bat.n.01"}, {"synset_id": "bat.n.02"}],
            },
            {
                "class_id": 2,
                "label": "dog",
                "status": "linked_auto",
                "synset_id": "dog.n.01",
                "candidates": [{"synset_id": "dog.n.01"}],
            },
        ]
    }
    outputs = _serialize(mapping, {"resolved_count": 1})
    text = outputs["wordnet_review.csv"].decode()
    assert "bat" in text and "bat.n.01" in text
    assert "dog" not in text
    assert json.loads(outputs["wordnet_coverage.json"])["resolved_count"] == 1


def test_atomic_idempotence_and_no_overwrite(tmp_path):
    output = {"mapping.json": b"one"}
    _save_outputs(tmp_path, output, dry_run=True)
    assert not (tmp_path / "mapping.json").exists()
    _save_outputs(tmp_path, output, dry_run=False)
    _save_outputs(tmp_path, output, dry_run=False)
    assert (tmp_path / "mapping.json").read_bytes() == b"one"
    with pytest.raises(FileExistsError):
        _save_outputs(tmp_path, {"mapping.json": b"two"}, dry_run=False)
    assert (tmp_path / "mapping.json").read_bytes() == b"one"
