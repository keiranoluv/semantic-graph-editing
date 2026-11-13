"""Unique lexical matching; replacement needs instruction and preserved context."""

from dataclasses import replace

from src.editing.operations import EditGraph
from tools.instructpix2pix_caption_graph_v1.instruction_check import matches


def align(source, target, request):
    left = {n.label: n for n in source.graph.nodes}
    right = {n.label: n for n in target.graph.nodes}
    if len(left) != len(source.graph.nodes) or len(right) != len(target.graph.nodes):
        raise ValueError("ambiguous_entity_alignment")
    mapping, evidence = {}, []
    for label in sorted(left.keys() & right.keys()):
        a, b = left[label], right[label]
        # Same lexical singleton, compatible role. "head" is compatible with
        # subject/object because adding/removing a relation changes those roles.
        ra, rb = source.evidence["roles"][a.id], target.evidence["roles"][b.id]
        if ra != rb and "head" not in (ra, rb):
            raise ValueError("entity_role_changed_or_ambiguous")
        mapping[b.id] = a.id
        evidence.append({"source": a.id, "target_local": b.id, "basis": "unique_normalized_label_and_compatible_role"})
    removed, added = [left[k] for k in left.keys() - right.keys()], [right[k] for k in right.keys() - left.keys()]
    if removed and added:
        if len(removed) != 1 or len(added) != 1 or request["kind"] != "replace":
            raise ValueError("ambiguous_or_unjustified_replacement")
        a, b = removed[0], added[0]
        if not matches(a, request["before"]) or not matches(b, request["after"]):
            raise ValueError("replacement_instruction_mismatch")
        if source.evidence["roles"][a.id] != target.evidence["roles"][b.id]:
            raise ValueError("replacement_syntactic_role_mismatch")
        mapping[b.id] = a.id
        evidence.append(
            {"source": a.id, "target_local": b.id, "basis": "unique_replacement_caption_delta_plus_instruction"}
        )
    for i, node in enumerate(n for n in target.graph.nodes if n.id not in mapping):
        mapping[node.id] = f"addedn{i+1:03d}"
    nodes = tuple(replace(n, id=mapping[n.id]) for n in target.graph.nodes)
    projected = [replace(e, subject=mapping[e.subject], object=mapping[e.object]) for e in target.graph.edges]
    source_edges = {(e.subject, e.object): e for e in source.graph.edges}
    if len(source_edges) != len(source.graph.edges) or len({(e.subject, e.object) for e in projected}) != len(
        projected
    ):
        raise ValueError("ambiguous_parallel_edge_alignment")
    edges = []
    for i, edge in enumerate(projected):
        old = source_edges.get((edge.subject, edge.object))
        edges.append(replace(edge, id=old.id if old else f"addede{i+1:03d}"))
    aligned = EditGraph(nodes, tuple(edges))
    # For a replacement all incident context must survive identically; a changed
    # predicate alongside object replacement is a compound edit, not guessed.
    if removed and added:
        a = removed[0]
        context_old = {(e.subject, e.predicate, e.object) for e in source.graph.edges if a.id in (e.subject, e.object)}
        context_new = {(e.subject, e.predicate, e.object) for e in aligned.edges if a.id in (e.subject, e.object)}
        if context_old != context_new:
            raise ValueError("replacement_relationship_context_changed")
    return (
        source.graph,
        aligned,
        {"nodes": evidence, "target_to_source_ids": mapping, "edges": "unique_directed_endpoint_pair"},
    )
