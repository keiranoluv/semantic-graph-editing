"""Deterministic observation-to-graph adapter; no image model or instruction heuristics."""

from src.editing.operations import Edge, EditGraph, Node, apply_operations, diff_aligned_graphs
from tools.edit_model_v1.codec import normalize_label
from tools.magicbrush_joint_v1.schema import validate


def text(value):
    return " ".join(value.casefold().split())


def indexed(rows):
    table = {r["id"]: r for r in rows}
    if len(table) != len(rows):
        raise ValueError("duplicate_observation_id")
    return table


def aligned(left, right, links, prefix):
    old, new, changed = {}, {}, []
    # Source identities depend only on source handles. No target-driven reordering.
    source_ids = {key: f"{prefix}{i+1:03d}" for i, key in enumerate(sorted(left))}
    additions = 0
    for link in sorted(links, key=lambda r: (r["source_id"] or "~", r["target_id"] or "~")):
        a, b = link["source_id"], link["target_id"]
        if a is None and b is None:
            raise ValueError("empty_correspondence")
        if link["certainty"] != "clear":
            raise ValueError("ambiguous_correspondence")
        if (a is not None and (a not in left or a in old)) or (b is not None and (b not in right or b in new)):
            raise ValueError("invalid_or_multiple_correspondence")
        if a is not None:
            identifier = source_ids[a]
            old[a] = identifier
        else:
            additions += 1
            identifier = f"new{prefix}{additions:03d}"
        if b is not None:
            new[b] = identifier
        changed.append((a, b))
    if old.keys() != left.keys() or new.keys() != right.keys():
        raise ValueError("missing_correspondence")
    return old, new, changed


def compile_annotation(annotation, limits):
    validate(annotation)
    source_nodes, target_nodes = (indexed(annotation[f"{role}_entities"]) for role in ("source", "target"))
    source_edges, target_edges = (indexed(annotation[f"{role}_relations"]) for role in ("source", "target"))
    left_nodes, right_nodes, node_links = aligned(source_nodes, target_nodes, annotation["entity_correspondences"], "n")
    left_edges, right_edges, edge_links = aligned(
        source_edges, target_edges, annotation["relation_correspondences"], "e"
    )

    def graph(nodes, edges, node_ids, edge_ids):
        if len(nodes) > limits["max_nodes"] or len(edges) > limits["max_edges"]:
            raise ValueError("graph_size_limit")
        # Keep duplicate attributes detectable instead of silently deduplicating them.
        try:
            return EditGraph(
                tuple(
                    Node(node_ids[k], text(r["label"]), tuple(text(a) for a in r["attributes"]))
                    for k, r in nodes.items()
                ),
                tuple(
                    Edge(edge_ids[k], node_ids[r["subject"]], text(r["predicate"]), node_ids[r["object"]])
                    for k, r in edges.items()
                ),
            )
        except KeyError as error:
            raise ValueError("dangling_relation_endpoint") from error

    source = graph(source_nodes, source_edges, left_nodes, left_edges)
    target = graph(target_nodes, target_edges, right_nodes, right_edges)
    expected, unchanged = set(), set()
    for kind, links, before, after, ids in (
        ("node", node_links, {n.id: n for n in source.nodes}, {n.id: n for n in target.nodes}, left_nodes),
        ("edge", edge_links, {e.id: e for e in source.edges}, {e.id: e for e in target.edges}, left_edges),
    ):
        for a, b in links:
            change = "addition" if a is None else "removal" if b is None else "replacement"
            differs = a is None or b is None or before[ids[a]] != after[ids[a]]
            if differs:
                expected.add((kind, a, b, change))
            elif kind == "node":
                unchanged.add(a)
    declared = [(c["entity_type"], c["source_id"], c["target_id"], c["change"]) for c in annotation["observed_changes"]]
    if len(declared) != len(set(declared)) or set(declared) != expected:
        raise ValueError("observed_changes_graph_mismatch")
    if (
        len(annotation["unchanged_entities"]) != len(set(annotation["unchanged_entities"]))
        or set(annotation["unchanged_entities"]) != unchanged
    ):
        raise ValueError("unexpected_modification_to_unchanged_entities")
    operations = diff_aligned_graphs(source, target)
    if len(operations) > limits["max_operations"]:
        raise ValueError("operation_count_limit")
    if apply_operations(source, operations) != target:
        raise ValueError("transaction_target_mismatch")
    # Reuse downstream source-only canonicalization; rejects indistinguishable new instances.
    source, target, operations = normalize_label(source, target, operations)
    if apply_operations(source, operations) != target:
        raise ValueError("normalized_transaction_target_mismatch")
    problems = []
    if annotation["annotation_status"] != "supported":
        problems.append("annotation_" + annotation["annotation_status"])
    if annotation["instruction_fulfilled"] != "yes":
        problems.append("instruction_" + annotation["instruction_fulfilled"])
    if annotation["uncertainties"]:
        problems.append("visual_uncertainties")
    if any(c["unintended"] for c in annotation["observed_changes"]):
        problems.append("unintended_changes")
    if not operations:
        problems.append("no_represented_change")
    classification = "needs_review" if problems else "candidate_accepted"
    if annotation["annotation_status"] == "unsupported":
        classification = "rejected"
    return {
        "source_graph": source.to_dict(),
        "target_graph": target.to_dict(),
        "operations": [op.to_dict() for op in operations],
        "status": "candidate" if classification == "candidate_accepted" else classification,
        "classification": classification,
        "problems": problems,
        "eligible_for_training": classification == "candidate_accepted",
        "human_review_status": "pending",
        "visual_reliability": "unverified",
        "model_estimated_confidence": annotation["model_confidence"],
        "target_reconstruction_exact": True,
        "validation": {"schema_valid": True, "graph_compiled": True, "transaction_valid": True},
        "annotation": annotation,
    }
