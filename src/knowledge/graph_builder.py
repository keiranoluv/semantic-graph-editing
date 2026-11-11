"""Leakage-safe, deterministic WordNet knowledge graph construction.

This module consumes ONLY external WordNet synsets and Milestone A's
vocabulary-to-synset mapping. No Visual Genome images or predicates are read.
"""

from collections import deque
from typing import Any

from src.knowledge.sense_review import _validate_mapping

# Direction, relation labels and WordNet source accessors are explicit.
WORDNET_RELATIONS = (
    ("IsA", "hypernyms", "wordnet.hypernyms"),
    ("InstanceOf", "instance_hypernyms", "wordnet.instance_hypernyms"),
    ("HasPart", "part_meronyms", "wordnet.part_meronyms"),
    ("HasMember", "member_meronyms", "wordnet.member_meronyms"),
    ("HasSubstance", "substance_meronyms", "wordnet.substance_meronyms"),
)


def build_wordnet_graph(
    mapping: dict,
    *,
    wordnet: Any,
    max_hops: int = 2,
    max_entities: int = 10000,
    max_triples: int = 50000,
    require_fully_reviewed: bool = False,
    reviewed_class_ids: frozenset[int] | None = None,
) -> dict:
    """Expand pinned mapped noun synsets into an auditable WordNet-only KG.

    Class-to-entity links are complete and ID-aligned. Unresolved classes map
    to None, not to a fabricated embedding. A draft KG may be created from
    partially resolved mappings, but is not flagged as training-ready.
    """
    if max_hops not in (1, 2, 3) or max_entities <= 0 or max_triples <= 0:
        raise ValueError("max_hops must be 1..3 and node/edge budgets positive")
    rows = _validate_mapping(mapping)
    if str(wordnet.get_version()) != str(mapping.get("source_version")):
        raise ValueError("WordNet version mismatch between source mapping and current corpus")
    unresolved = [r["class_id"] for r in rows[1:] if r["status"] in {"ambiguous", "unmatched"}]
    reviewed = reviewed_class_ids if reviewed_class_ids is not None else frozenset()
    unreviewed = [
        r["class_id"]
        for r in rows[1:]
        if r["status"] in {"linked_auto", "linked_manual"} and r["class_id"] not in reviewed
    ]
    if require_fully_reviewed and (unresolved or unreviewed):
        raise ValueError("Review gate failed: ambiguous/unmatched/unaudited classes remain")

    class_synsets = []
    roots = set()
    for row in rows:
        name = row.get("synset_id") if row["class_id"] > 0 else None
        if row["class_id"] > 0 and row["status"] in {"linked_auto", "linked_manual"}:
            if not isinstance(name, str) or not name:
                raise ValueError(f"Missing linked synset at ID {row['class_id']}")
            synset = wordnet.synset(name)
            if synset.name() != name or synset.pos() != "n":
                raise ValueError(f"Invalid WordNet noun at ID {row['class_id']}")
            roots.add(name)
        elif name is not None:
            raise ValueError("Unresolved/background class must not point to a synset")
        class_synsets.append(name)
    if not roots:
        raise ValueError("No resolved WordNet roots; review a mapping first")

    discovered = set(roots)
    visited = set()
    # Start with all roots at hop zero to ensure deterministic shortest-hop expansion.
    queue = deque((name, 0) for name in sorted(roots))
    triples = set()
    while queue:
        name, depth = queue.popleft()
        if name in visited:
            continue
        visited.add(name)
        if depth >= max_hops:
            continue
        synset = wordnet.synset(name)
        neighbors = []
        for relation, method, provenance in WORDNET_RELATIONS:
            for neighbor in getattr(synset, method)():
                if neighbor.pos() != "n":
                    continue
                neighbor_name = neighbor.name()
                neighbors.append((relation, neighbor_name, provenance))
        # Sorted traversal and sorted export ensure stable outputs across runs.
        for relation, target, provenance in sorted(neighbors):
            triples.add((name, relation, target, provenance))
            if len(triples) > max_triples:
                raise ValueError("KG exceeds max_triples; increase the explicit safety limit")
            if target not in discovered:
                discovered.add(target)
                if len(discovered) > max_entities:
                    raise ValueError("KG exceeds max_entities; increase the explicit safety limit")
                queue.append((target, depth + 1))

    entities = [f"wn:{synset}" for synset in sorted(discovered)]
    entity_to_id = {entity: i for i, entity in enumerate(entities)}
    relations = [name for name, _, _ in WORDNET_RELATIONS]
    relation_to_id = {relation: i for i, relation in enumerate(relations)}
    encoded = []
    for source, relation, target, provenance in sorted(triples):
        encoded.append(
            {
                "subject_id": entity_to_id[f"wn:{source}"],
                "relation_id": relation_to_id[relation],
                "object_id": entity_to_id[f"wn:{target}"],
                "provenance": provenance,
            }
        )
    class_entity_ids = [entity_to_id[f"wn:{name}"] if name else None for name in class_synsets]
    return {
        "schema_version": 1,
        "source": "wordnet",
        "source_version": mapping["source_version"],
        "entities": entities,
        "relations": relations,
        "triples": encoded,
        "class_entity_ids": class_entity_ids,
        "metadata": {
            "max_hops": max_hops,
            "max_entities": max_entities,
            "max_triples": max_triples,
            "root_count": len(roots),
            "entity_count": len(entities),
            "triple_count": len(encoded),
            "class_count_including_background": len(class_synsets),
            "resolved_class_count": len(rows) - 1 - len(unresolved),
            "unresolved_class_ids": unresolved,
            "unreviewed_class_ids": unreviewed,
            "ready_for_gat_kg_training": not unresolved and not unreviewed,
            "leakage_policy": "external_wordnet_only_no_vg_predicate_annotations",
        },
    }
