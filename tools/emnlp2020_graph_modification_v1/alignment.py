"""Conservative exhaustive alignment; raw integer equality carries no evidence."""

from itertools import permutations

from src.editing.operations import Edge, EditGraph, Node
from tools.emnlp2020_graph_modification_v1.dataset import ALIGNMENT


class AlignmentError(ValueError):
    pass


def validate_raw(raw):
    if not isinstance(raw, dict) or set(raw) != {"directed", "nodes", "edges", "graph_metadata"}:
        raise ValueError("unknown_raw_schema")
    if type(raw["directed"]) is not bool or raw["graph_metadata"] != {}:
        raise ValueError("unknown_graph_metadata")
    nodes = {}
    for row in raw["nodes"]:
        if set(row) != {"id", "data"} or type(row["id"]) is not int or set(row["data"]) != {"feature"}:
            raise ValueError("unknown_node_schema")
        identifier, label = row["id"], row["data"]["feature"]
        if identifier in nodes or not isinstance(label, str) or not label.strip():
            raise ValueError("duplicate_or_invalid_node")
        nodes[identifier] = label
    edges = {}
    for row in raw["edges"]:
        if set(row) != {"u", "v", "data"} or set(row["data"]) != {"feature"}:
            raise ValueError("unknown_edge_schema")
        u, v, label = row["u"], row["v"], row["data"]["feature"]
        if u not in nodes or v not in nodes or not isinstance(label, str) or not label.strip():
            raise ValueError("dangling_or_invalid_edge")
        key = (u, v) if raw["directed"] else tuple(sorted((u, v)))
        if key in edges:
            raise ValueError("duplicate_semantic_or_parallel_edge")
        edges[key] = label
    return nodes, edges


def attribute_policy(raw, policy="A"):
    """B is evaluated but refuses to invent ownership absent in this release.

    Undirected ``att`` links do not say which endpoint is the owner. Even a
    degree-one endpoint is insufficient (both endpoints can be leaves). Node
    features have no object/attribute type. No identities are discarded.
    """
    validate_raw(raw)
    if policy not in ("A", "B"):
        raise ValueError("unknown_attribute_policy")
    att = [e for e in raw["edges"] if e["data"]["feature"] == "att"]
    if policy == "B" and att:
        raise ValueError("attribute_ownership_unproven_refuse_flattening")
    return raw, {"policy": policy, "attribute_links": len(att), "lossy": False, "transformations": []}


def canonical_raw(raw):
    nodes, edges = validate_raw(raw)
    node_ids = {identifier: f"n{i+1:03d}" for i, identifier in enumerate(sorted(nodes))}
    triples = []
    for (u, v), predicate in edges.items():
        triples.append((node_ids[u], predicate, node_ids[v]))
        if not raw["directed"] and u != v:
            triples.append((node_ids[v], predicate, node_ids[u]))
    graph = EditGraph(
        tuple(Node(node_ids[k], label) for k, label in nodes.items()),
        tuple(Edge(f"e{i+1:03d}", u, p, v) for i, (u, p, v) in enumerate(sorted(triples))),
    )
    verify_topology(raw, graph, node_ids)
    return graph, node_ids


def verify_topology(raw, graph, ids):
    """Exact information-preservation check relative to the stored graph."""
    nodes, edges = validate_raw(raw)
    if len(set(ids.values())) != len(ids) or {n.id: n.label for n in graph.nodes} != {
        ids[k]: v for k, v in nodes.items()
    }:
        raise ValueError("node_conversion_not_lossless")
    if any(n.attributes for n in graph.nodes):
        raise ValueError("unexpected_flattened_attributes")
    expected = set()
    for (u, v), p in edges.items():
        expected.add((ids[u], p, ids[v]))
        if not raw["directed"]:
            expected.add((ids[v], p, ids[u]))
    if {(e.subject, e.predicate, e.object) for e in graph.edges} != expected:
        raise ValueError("edge_conversion_not_lossless")


def align(source_raw, target_raw):
    left, le = validate_raw(source_raw)
    right, re = validate_raw(target_raw)
    if source_raw["directed"] != target_raw["directed"]:
        raise AlignmentError("directedness_changed")
    if max(len(left), len(right)) > 7:
        raise AlignmentError("alignment_search_budget_exceeded")
    # Real records also remove an object together with its attribute subtree.
    # Allow multiple unmatched nodes, but require labels and ALL retained
    # relations to be preserved on insert/delete. Substitution changes one label.
    directed = source_raw["directed"]
    ls, rs = sorted(left), sorted(right)
    if len(left) <= len(right):
        possibilities = ({t: s for s, t in zip(ls, order, strict=True)} for order in permutations(rs, len(ls)))
    else:
        possibilities = (dict(zip(rs, order, strict=True)) for order in permutations(ls, len(rs)))

    def key(u, v):
        return (u, v) if directed else tuple(sorted((u, v)))

    best, winners = None, []
    for mapping in possibilities:
        changes = sum(left[s] != right[t] for t, s in mapping.items())
        if changes > (1 if len(ls) == len(rs) else 0):
            continue
        mapped_edges = {key(mapping[u], mapping[v]): p for (u, v), p in re.items() if u in mapping and v in mapping}
        retained = set(mapping.values())
        retained_edges = {(u, v): p for (u, v), p in le.items() if u in retained and v in retained}
        edge_changes = sum(
            retained_edges.get(k) != mapped_edges.get(k) for k in retained_edges.keys() | mapped_edges.keys()
        )
        score = (changes, edge_changes)
        if best is None or score < best:
            best, winners = score, [mapping]
        elif score == best:
            winners.append(mapping)
    if not winners:
        raise AlignmentError("no_constrained_node_mapping")
    if len(winners) != 1:
        raise AlignmentError(f"ambiguous_node_identity:{len(winners)}")
    mapping = winners[0]
    if best[0] and best[1]:
        raise AlignmentError("substitution_topology_changed")
    if len(ls) != len(rs) and best[1]:
        raise AlignmentError("insertion_deletion_changed_retained_edges")
    source, source_ids = canonical_raw(source_raw)
    target_ids = {t: source_ids[s] for t, s in mapping.items()}
    for i, t in enumerate(sorted(right.keys() - mapping.keys())):
        target_ids[t] = f"rawnewn{i:03d}"
    old_triples = {(e.subject, e.predicate, e.object): e.id for e in source.edges}
    old_pairs = {}
    for edge in source.edges:
        old_pairs.setdefault((edge.subject, edge.object), []).append(edge)
    target_triples = []
    for (u, v), predicate in re.items():
        target_triples.append((target_ids[u], predicate, target_ids[v]))
        if not directed and u != v:
            target_triples.append((target_ids[v], predicate, target_ids[u]))
    target_edges, correspondence = [], []
    for i, triple in enumerate(sorted(target_triples)):
        u, p, v = triple
        old_id = old_triples.get(triple)
        basis = "unchanged_semantic_triple" if old_id else "new_edge"
        if old_id is None:
            candidates = old_pairs.get((u, v), [])
            new_same_pair = [x for x in target_triples if (x[0], x[2]) == (u, v)]
            if len(candidates) == len(new_same_pair) == 1:
                old_id, basis = candidates[0].id, "unique_ordered_endpoint_pair_relation_replacement"
        identifier = old_id or f"rawnewe{i:03d}"
        target_edges.append(Edge(identifier, u, p, v))
        correspondence.append({"target_triple": triple, "source_edge_id": old_id, "basis": basis})
    target = EditGraph(tuple(Node(target_ids[k], label) for k, label in right.items()), tuple(target_edges))
    verify_topology(source_raw, source, source_ids)
    verify_topology(target_raw, target, target_ids)
    evidence = {
        "algorithm": ALIGNMENT,
        "source_ids_used_as_cross_graph_evidence": False,
        "score": best,
        "optimal_mappings": 1,
        "node_correspondence": [
            {"target_raw_id": t, "source_raw_id": s, "canonical_source_id": source_ids[s]}
            for t, s in sorted(mapping.items())
        ],
        "added_target_raw_ids": sorted(right.keys() - mapping.keys()),
        "deleted_source_raw_ids": sorted(left.keys() - set(mapping.values())),
        "edge_correspondence_before_new_id_normalization": correspondence,
        "source_node_ids": {str(k): v for k, v in source_ids.items()},
        "target_node_ids_before_new_id_normalization": {str(k): v for k, v in target_ids.items()},
        "semantics": "directed_as_stored" if directed else "undirected_each_link_represented_by_reciprocal_arcs",
        "confidence": "unique_constrained_structural_mapping_not_verified_historical_identity",
        "exact_stored_topology_preservation_verified": True,
    }
    return source, target, evidence
