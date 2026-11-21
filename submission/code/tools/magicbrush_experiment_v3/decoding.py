"""Source-only graph construction with auditable structural constraints.

Raw head decisions are retained and scored separately. The constrained output
masks dead endpoints, explicitly deletes edges incident to removed nodes, and
keeps the first edge for duplicate triples. It never consults a reference target.
"""
from collections import Counter

import torch

from src.editing.operations import Edge, EditGraph, Node, Operation, diff_aligned_graphs
from tools.edit_model_v1.codec import encode_actions


def decode_batch(inputs, logits, vocab, max_nodes):
    # Moving small categorical logits to CPU once avoids per-field GPU syncs.
    logits = {k: v.detach().cpu() for k, v in logits.items()}
    return [decode_one(item, logits, vocab, max_nodes, b) for b, item in enumerate(inputs)]


def decode_one(item, logits, vocab, max_nodes, b):
    source = item["source"]
    flags, raw_ops = Counter(), []

    def choice(key, slot):
        return int(logits[key][b, slot].argmax())

    def semantic(prefix, field, slot, before=None):
        index = choice(prefix + "_" + field, slot)
        if index == 0:
            if before is None:
                raise ValueError("New entity selected impossible COPY category")
            return before
        values = vocab[{"label": "labels", "attributes": "attributes", "predicate": "predicates"}[field]]
        value = values[index - 1]
        return tuple(value) if field == "attributes" else value

    def attributes(prefix, slot, before=()):
        remove = logits[prefix + "_remove_attributes"][b, slot]
        retained = {value for i, value in enumerate(before) if i >= len(remove) or remove[i] <= 0}
        count = choice(prefix + "_attribute_count", slot)
        ranked = logits[prefix + "_add_attributes"][b, slot].argsort(descending=True, stable=True).tolist()
        candidates = [vocab["attributes"][i] for i in ranked if vocab["attributes"][i] not in before]
        return tuple(sorted(retained | set(candidates[:count])))

    def append_operation(entity, before, after):
        if before == after:
            return
        kind = "add" if before is None else "delete" if after is None else "replace"
        raw_ops.append(Operation(kind + "_" + entity, before, after))

    nodes = []
    for i, old in enumerate(source.nodes):
        action = choice("node_action", i)
        after = (
            old
            if action == 0
            else None
            if action == 1
            else Node(old.id, semantic("node", "label", i, old.label), attributes("node", i, old.attributes))
        )
        append_operation("node", old, after)
        if after is not None:
            nodes.append(after)
    added_count = logits["new_node_present"].shape[1]
    for i in range(added_count):
        if choice("new_node_present", i):
            node = Node(f"newn{i+1:03d}", semantic("new_node", "label", i), attributes("new_node", i))
            nodes.append(node)
            append_operation("node", None, node)
    alive = {n.id for n in nodes}
    slots = [n.id for n in source.nodes] + [f"absent{i:03d}" for i in range(len(source.nodes), max_nodes)]
    slots += [f"newn{i+1:03d}" for i in range(added_count)]

    def endpoint(prefix, field, i, before, constrained):
        scores = logits[prefix + "_" + field][b, i].clone()
        identifiers = [before, *slots]
        if constrained:
            mask = torch.tensor([x in alive for x in identifiers], dtype=torch.bool)
            if not mask.any():
                return None
            scores[~mask] = -torch.inf
        index = int(scores.argmax())
        if identifiers[index] is None:
            raise ValueError("New edge selected impossible COPY endpoint")
        if constrained and index != choice(prefix + "_" + field, i):
            flags["masked_dead_endpoint_selections"] += 1
        return identifiers[index]

    edges, triples = [], set()

    def retain(edge):
        triple = (edge.subject, edge.predicate, edge.object)
        if triple in triples:
            flags["duplicate_edges_removed"] += 1
            return
        triples.add(triple)
        edges.append(edge)

    def make_edge(prefix, i, ident, before=None, constrained=False):
        subject = endpoint(prefix, "subject", i, before.subject if before else None, constrained)
        obj = endpoint(prefix, "object", i, before.object if before else None, constrained)
        if subject is None or obj is None:
            return None
        return Edge(ident, subject, semantic(prefix, "predicate", i, before.predicate if before else None), obj)

    for i, old in enumerate(source.edges):
        action = choice("edge_action", i)
        raw = old if action == 0 else None if action == 1 else make_edge("edge", i, old.id, old)
        append_operation("edge", old, raw)
        if action == 0:
            if old.subject in alive and old.object in alive:
                retain(old)
            else:
                flags["incident_edges_explicitly_deleted"] += 1
        elif action == 2:
            edge = make_edge("edge", i, old.id, old, constrained=True)
            if edge is not None:
                retain(edge)
            else:
                flags["edges_removed_without_alive_nodes"] += 1
    for i in range(logits["new_edge_present"].shape[1]):
        if choice("new_edge_present", i):
            ident = f"newe{i+1:03d}"
            append_operation("edge", None, make_edge("new_edge", i, ident))
            edge = make_edge("new_edge", i, ident, constrained=True)
            if edge is not None:
                retain(edge)
            else:
                flags["edges_removed_without_alive_nodes"] += 1
    target = EditGraph(tuple(nodes), tuple(edges))
    actions = diff_aligned_graphs(source, target)
    # Do not silently prune >12 actions or merge ambiguous new instances.
    # The unchanged evaluator counts those predictions as failures.
    return {
        "tokens": encode_actions(actions)[1:],
        "raw_tokens": encode_actions(raw_ops)[1:],
        "constraints": dict(flags),
        "operation_count": len(actions),
    }
