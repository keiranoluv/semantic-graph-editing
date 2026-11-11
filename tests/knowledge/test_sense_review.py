"""A2 review tests: synthetic WordNet, no VG150 asset or downloaded corpus."""

import csv
import io

import pytest

from src.knowledge.sense_review import compile_reviews, rank_mapping, review_csv


class Lemma:
    def __init__(self, name, count=0):
        self._name, self._count = name, count

    def name(self):
        return self._name

    def count(self):
        return self._count


class Synset:
    def __init__(self, name, lemmas):
        self._name, self._lemmas = name, lemmas

    def name(self):
        return self._name

    def pos(self):
        return "n"

    def lemmas(self):
        return self._lemmas


class WordNet:
    def __init__(self):
        self.synsets = {
            "bicycle.n.01": Synset("bicycle.n.01", [Lemma("bike", 15)]),
            "motorcycle.n.01": Synset("motorcycle.n.01", [Lemma("bike", 7)]),
            "dog.n.01": Synset("dog.n.01", [Lemma("dog", 21)]),
        }

    def synset(self, name):
        return self.synsets[name]

    def get_version(self):
        return "test-wn"


def mapping():
    return {
        "schema_version": 1,
        "source": "wordnet",
        "source_version": "test-wn",
        "rows": [
            {"class_id": 0, "label": "__background__", "status": "background", "synset_id": None},
            {
                "class_id": 1,
                "label": "bike",
                "normalized_label": "bike",
                "status": "ambiguous",
                "synset_id": None,
                "candidates": [
                    {"synset_id": "bicycle.n.01", "definition": "pedal cycle"},
                    {"synset_id": "motorcycle.n.01", "definition": "motorized bike"},
                ],
            },
            {
                "class_id": 2,
                "label": "dog",
                "normalized_label": "dog",
                "status": "linked_auto",
                "synset_id": "dog.n.01",
                "candidates": [{"synset_id": "dog.n.01", "definition": "domestic canine"}],
            },
        ],
    }


def edit_review(data, changes):
    rows = list(csv.DictReader(io.StringIO(data.decode())))
    for row in rows:
        row.update(changes.get(row["label"], {}))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def test_rank_determinism_and_reviewer_gate():
    rank1 = rank_mapping(mapping(), wordnet=WordNet())
    assert rank1 == rank_mapping(mapping(), wordnet=WordNet())
    assert rank1["rows"][0]["suggested_synset"] == "bicycle.n.01"
    data = review_csv(rank1)
    overrides, report = compile_reviews(mapping(), data, {"version": 1, "mappings": {}})
    assert overrides["mappings"] == {}
    assert report["accepted_decisions"] == 0


def test_accepted_override_requires_explicit_decision_and_notes():
    rank = rank_mapping(mapping(), wordnet=WordNet())
    data = edit_review(
        review_csv(rank),
        {
            "bike": {
                "decision": "accept",
                "selected_synset": "motorcycle.n.01",
                "reviewer": "human-reviewer",
                "notes": "Manual assessment",
            }
        },
    )
    overrides, report = compile_reviews(mapping(), data, {"version": 1, "mappings": {}})
    assert overrides["mappings"] == {"bike": "motorcycle.n.01"}
    assert overrides["review_provenance"]["bike"]["notes"] == "Manual assessment"
    assert report["accepted_decisions"] == 1


def test_invalid_or_unguarded_review_rejected():
    data = review_csv(rank_mapping(mapping(), wordnet=WordNet()))
    with pytest.raises(ValueError, match="not in source candidates"):
        compile_reviews(
            mapping(),
            edit_review(
                data,
                {"bike": {"decision": "accept", "selected_synset": "cat.n.01", "reviewer": "alice", "notes": "oops"}},
            ),
            {"version": 1, "mappings": {}},
        )
    with pytest.raises(ValueError, match="reviewer identity required"):
        compile_reviews(
            mapping(),
            edit_review(data, {"bike": {"decision": "accept", "selected_synset": "bicycle.n.01", "notes": "checked"}}),
            {"version": 1, "mappings": {}},
        )
    with pytest.raises(ValueError, match="rationale required"):
        compile_reviews(
            mapping(),
            edit_review(data, {"bike": {"decision": "accept", "selected_synset": "bicycle.n.01", "reviewer": "alice"}}),
            {"version": 1, "mappings": {}},
        )


def test_row_tampering_rejected():
    data = review_csv(rank_mapping(mapping(), wordnet=WordNet()))
    with pytest.raises(ValueError, match="alignment mismatch"):
        compile_reviews(mapping(), edit_review(data, {"bike": {"class_id": "99"}}), {"version": 1, "mappings": {}})


def test_version_mismatch_rejected():
    bad = mapping()
    bad["source_version"] = "other"
    with pytest.raises(ValueError, match="version mismatch"):
        rank_mapping(bad, wordnet=WordNet())
