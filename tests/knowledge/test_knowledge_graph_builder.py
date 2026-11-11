"""B1 KG topology, aligned IDs and leakage properties using toy WordNet."""

import json

import pytest

from scripts.build_wordnet_graph import serialize_graph
from src.knowledge.graph_builder import build_wordnet_graph


class Synset:
    def __init__(self, name):
        self._name = name
        self._relations = {}

    def name(self):
        return self._name

    def pos(self):
        return "n"

    def hypernyms(self):
        return self._relations.get("hypernyms", [])

    def instance_hypernyms(self):
        return self._relations.get("instance_hypernyms", [])

    def part_meronyms(self):
        return self._relations.get("part_meronyms", [])

    def member_meronyms(self):
        return self._relations.get("member_meronyms", [])

    def substance_meronyms(self):
        return self._relations.get("substance_meronyms", [])


class WordNet:
    def __init__(self):
        self.nodes = {name: Synset(name) for name in ("dog.n.01", "canine.n.02", "animal.n.01", "tail.n.01")}
        self.nodes["dog.n.01"]._relations["hypernyms"] = [self.nodes["canine.n.02"]]
        self.nodes["dog.n.01"]._relations["part_meronyms"] = [self.nodes["tail.n.01"]]
        self.nodes["canine.n.02"]._relations["hypernyms"] = [self.nodes["animal.n.01"]]

    def synset(self, name):
        return self.nodes[name]

    def get_version(self):
        return "toy-version"


def mapping():
    return {
        "schema_version": 1,
        "source": "wordnet",
        "source_version": "toy-version",
        "rows": [
            {"class_id": 0, "label": "__background__", "status": "background", "synset_id": None},
            {"class_id": 1, "label": "dog", "status": "linked_manual", "synset_id": "dog.n.01", "candidates": []},
            {"class_id": 2, "label": "hound", "status": "linked_auto", "synset_id": "dog.n.01", "candidates": []},
            {"class_id": 3, "label": "unknown", "status": "ambiguous", "synset_id": None, "candidates": []},
        ],
    }


def test_b1_links_and_direction_and_two_hops():
    g = build_wordnet_graph(mapping(), wordnet=WordNet(), max_hops=2)
    assert g == build_wordnet_graph(mapping(), wordnet=WordNet(), max_hops=2)
    assert g["class_entity_ids"][0] is None
    assert g["class_entity_ids"][1] == g["class_entity_ids"][2]
    assert g["class_entity_ids"][3] is None
    triples = {
        (g["entities"][e["subject_id"]], g["relations"][e["relation_id"]], g["entities"][e["object_id"]])
        for e in g["triples"]
    }
    assert ("wn:dog.n.01", "IsA", "wn:canine.n.02") in triples
    assert ("wn:dog.n.01", "HasPart", "wn:tail.n.01") in triples
    assert ("wn:canine.n.02", "IsA", "wn:animal.n.01") in triples
    assert g["metadata"]["ready_for_gat_kg_training"] is False
    assert g["metadata"]["unresolved_class_ids"] == [3]


def test_one_hop_not_two_hops():
    g = build_wordnet_graph(mapping(), wordnet=WordNet(), max_hops=1)
    assert "wn:animal.n.01" not in g["entities"]
    assert len(g["triples"]) == 2


def test_review_gate_and_limits():
    with pytest.raises(ValueError, match="Review gate"):
        build_wordnet_graph(mapping(), wordnet=WordNet(), require_fully_reviewed=True)
    with pytest.raises(ValueError, match="max_entities"):
        build_wordnet_graph(mapping(), wordnet=WordNet(), max_entities=2)
    with pytest.raises(ValueError, match="max_triples"):
        build_wordnet_graph(mapping(), wordnet=WordNet(), max_triples=1)


def test_rejects_unsupported_source_version_and_missing_roots():
    bad = mapping()
    bad["source_version"] = "wrong"
    with pytest.raises(ValueError, match="version mismatch"):
        build_wordnet_graph(bad, wordnet=WordNet())
    bad = mapping()
    for row in bad["rows"][1:]:
        row["status"] = "unmatched"
        row["synset_id"] = None
    with pytest.raises(ValueError, match="No resolved"):
        build_wordnet_graph(bad, wordnet=WordNet())


def test_export_id_integrity_and_provenance():
    g = build_wordnet_graph(mapping(), wordnet=WordNet())
    output = serialize_graph(g, mapping_file_sha256="0123")
    metadata = json.loads(output["metadata.json"])
    assert metadata["input_mapping_sha256"] == "0123"
    assert len(json.loads(output["class_entity_ids.json"])["class_entity_ids"]) == 4
    assert output["triples.tsv"].decode().count("wordnet.") == len(g["triples"])


def test_full_audit_provenance_gate():
    fully_linked = mapping()
    fully_linked["rows"][2]["status"] = "linked_manual"
    fully_linked["rows"][3]["status"] = "linked_manual"
    fully_linked["rows"][3]["synset_id"] = "tail.n.01"
    graph = build_wordnet_graph(
        fully_linked,
        wordnet=WordNet(),
        reviewed_class_ids=frozenset({1, 2, 3}),
        require_fully_reviewed=True,
    )
    assert graph["metadata"]["ready_for_gat_kg_training"] is True
    assert not graph["metadata"]["unreviewed_class_ids"]
    with pytest.raises(ValueError, match="Review gate"):
        build_wordnet_graph(
            fully_linked, wordnet=WordNet(), reviewed_class_ids=frozenset({1, 2}), require_fully_reviewed=True
        )
