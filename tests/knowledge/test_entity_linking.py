"""Synthetic, corpus-independent WordNet linking tests (no VG150 data)."""

import json

import pytest

from src.knowledge.entity_linking import link_vocabulary


class FakeSynset:
    def __init__(self, name, description="a noun"):
        self._name = name
        self._definition = description

    def name(self):
        return self._name

    def definition(self):
        return self._definition

    def pos(self):
        return "n"


class FakeWordNet:
    def __init__(self):
        self._senses = {
            "traffic_light": [FakeSynset("traffic_light.n.01")],
            "bat": [FakeSynset("bat.n.01"), FakeSynset("bat.n.02")],
            "dog": [FakeSynset("dog.n.01")],
        }

    def synsets(self, name, pos):
        assert pos == "n"
        return self._senses.get(name, [])

    def morphy(self, name, pos):
        return None

    def synset(self, name):
        for values in self._senses.values():
            for synset in values:
                if synset.name() == name:
                    return synset
        raise ValueError("Unknown synset")

    def get_version(self):
        return "test-wordnet"


def _vocab(tmp_path):
    path = tmp_path / "vocab.json"
    path.write_text(
        json.dumps(
            {
                "label_to_idx": {
                    "dog": 3,
                    "bat": 2,
                    "traffic light": 1,
                    "__background__": 0,
                },
                "idx_to_label": {"1": "traffic light", "2": "bat", "3": "dog"},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_id_alignment_and_ambiguity(tmp_path):
    mapping, report = link_vocabulary(_vocab(tmp_path), wordnet=FakeWordNet(), expected_classes=3)
    assert [row["class_id"] for row in mapping["rows"]] == [0, 1, 2, 3]
    assert [row["status"] for row in mapping["rows"]] == ["background", "linked_auto", "ambiguous", "linked_auto"]
    assert mapping["rows"][2]["synset_id"] is None
    assert report["resolved_count"] == 2
    assert report["candidate_coverage"] == 1.0
    assert report["resolved_coverage"] == 2 / 3


def test_manual_override_and_provenance(tmp_path):
    path = tmp_path / "overrides.json"
    path.write_text(json.dumps({"version": 1, "mappings": {"bat": "bat.n.02"}}), encoding="utf-8")
    mapping, report = link_vocabulary(_vocab(tmp_path), wordnet=FakeWordNet(), expected_classes=3, overrides_path=path)
    assert mapping["rows"][2]["synset_id"] == "bat.n.02"
    assert mapping["rows"][2]["status"] == "linked_manual"
    assert mapping["rows"][2]["match_rule"] == "curated_override"
    assert len(mapping["overrides_sha256"]) == 64
    assert report["resolved_coverage"] == 1.0


def test_unknown_override_rejected(tmp_path):
    path = tmp_path / "overrides.json"
    path.write_text(json.dumps({"version": 1, "mappings": {"cat": "cat.n.01"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid override"):
        link_vocabulary(_vocab(tmp_path), wordnet=FakeWordNet(), expected_classes=3, overrides_path=path)


def test_misaligned_vocabulary_rejected(tmp_path):
    path = _vocab(tmp_path)
    data = json.loads(path.read_text())
    data["label_to_idx"]["dog"] = 4
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly IDs"):
        link_vocabulary(path, wordnet=FakeWordNet(), expected_classes=3)


def test_deterministic(tmp_path):
    args = {"wordnet": FakeWordNet(), "expected_classes": 3}
    first = link_vocabulary(_vocab(tmp_path), **args)
    second = link_vocabulary(_vocab(tmp_path), **args)
    assert first == second
