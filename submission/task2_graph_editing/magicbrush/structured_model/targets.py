"""Supervision for existing graph entities and canonical added-entity slots."""

import torch
from task2_graph_editing.magicbrush.structured_model.vocabulary import field_index
from torch.nn import functional as F


def encode_targets(examples, vocab, config, device="cpu"):
    arch = config["model"]
    n, m, a = arch["max_nodes"], arch["max_edges"], arch["add_slots"]
    shapes = {
        "node": (n, ("action", "label", "attribute_count")),
        "edge": (m, ("action", "predicate", "subject", "object")),
        "new_node": (a, ("present", "label", "attribute_count")),
        "new_edge": (a, ("present", "predicate", "subject", "object")),
    }
    result = {
        f"{prefix}_{field}": torch.full((len(examples), size), -100, dtype=torch.long, device=device)
        for prefix, (size, fields) in shapes.items()
        for field in fields
    }
    for prefix, size in (("node", n), ("new_node", a)):
        result[prefix + "_remove_attributes"] = torch.full(
            (len(examples), size, arch["max_attributes"]), -100.0, device=device
        )
        result[prefix + "_add_attributes"] = torch.full(
            (len(examples), size, len(vocab["attributes"])), -100.0, device=device
        )
    for b, e in enumerate(examples):
        old_nodes = {x.id: i for i, x in enumerate(e.source.nodes)}
        old_edges = {x.id: i for i, x in enumerate(e.source.edges)}
        added_nodes = sorted((x for x in e.target.nodes if x.id not in old_nodes), key=lambda x: x.id)
        added_edges = sorted((x for x in e.target.edges if x.id not in old_edges), key=lambda x: x.id)
        if len(added_nodes) > a or len(added_edges) > a:
            raise ValueError("Added entity count exceeds predeclared slots")
        node_slots = {**old_nodes, **{x.id: n + i for i, x in enumerate(added_nodes)}}
        result["node_action"][b, : len(old_nodes)] = 0
        result["edge_action"][b, : len(old_edges)] = 0
        result["new_node_present"][b] = 0
        result["new_edge_present"][b] = 0
        for op in e.operations:
            action, entity = op.kind.split("_")
            if action == "add":
                prefix = "new_" + entity
                items = added_nodes if entity == "node" else added_edges
                i = next(i for i, x in enumerate(items) if x.id == op.entity_id)
                result[prefix + "_present"][b, i] = 1
            else:
                prefix = entity
                i = (old_nodes if entity == "node" else old_edges)[op.entity_id]
                result[prefix + "_action"][b, i] = 1 if action == "delete" else 2
            if op.after is None:
                continue
            fields = (("label", "labels"),) if entity == "node" else (("predicate", "predicates"),)
            for field, vocab_field in fields:
                before = getattr(op.before, field) if op.before else None
                result[prefix + "_" + field][b, i] = field_index(vocab, vocab_field, getattr(op.after, field), before)
            if entity == "node":
                old_attrs = op.before.attributes if op.before else ()
                added_attrs = set(op.after.attributes) - set(old_attrs)
                if len(old_attrs) > arch["max_attributes"] or len(added_attrs) > arch["max_attributes"]:
                    raise ValueError("Attribute payload exceeds predeclared slot budget")
                result[prefix + "_attribute_count"][b, i] = len(added_attrs)
                for j, value in enumerate(old_attrs):
                    result[prefix + "_remove_attributes"][b, i, j] = float(value not in op.after.attributes)
                result[prefix + "_add_attributes"][b, i] = 0.0
                for value in added_attrs:
                    if value in vocab["attributes"]:
                        result[prefix + "_add_attributes"][b, i, vocab["attributes"].index(value)] = 1.0
            if entity == "edge":
                for field in ("subject", "object"):
                    value = getattr(op.after, field)
                    old = getattr(op.before, field) if op.before else None
                    result[prefix + "_" + field][b, i] = 0 if value == old else node_slots[value] + 1
    return result


def structured_loss(logits, targets, config):
    losses = []
    for field, values in logits.items():
        expected = targets[field]
        selected = expected != -100
        if not selected.any():
            continue
        if field.endswith("_attributes"):
            # Balance positive/negative decisions independently: thousands of
            # absent atoms must not drown out a few required attribute edits.
            errors = F.binary_cross_entropy_with_logits(values[selected], expected[selected], reduction="none")
            positives = expected[selected] > 0
            terms = [errors[mask].mean() for mask in (positives, ~positives) if mask.any()]
            losses.append(torch.stack(terms).mean())
            continue
        weights, scale = None, 1.0
        if field.endswith("_action"):
            weights = values.new_tensor(config["action_class_weights"])
            scale = config["loss_action_weight"]
        elif field.endswith("_present"):
            weights = values.new_tensor(config["presence_class_weights"])
            scale = config["loss_action_weight"]
        losses.append(scale * F.cross_entropy(values[selected], expected[selected], weight=weights))
    if not losses:
        raise ValueError("Empty structured supervision")
    return torch.stack(losses).mean()
