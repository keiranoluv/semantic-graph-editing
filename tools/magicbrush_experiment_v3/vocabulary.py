"""Frozen train-only semantic values and source-copy targets for graph slots."""
from dataclasses import dataclass

import torch

from src.editing.operations import EditGraph, Node
from tools.edit_model_v1.data import feature_texts


@dataclass(frozen=True)
class PrototypeInput:
    id: str
    source: EditGraph
    instruction: str = ""

    def model_input(self):
        return {"source": self.source, "instruction": self.instruction}


def build_vocabulary(train):
    if not train or any(e.split != "train" for e in train):
        raise ValueError("Vocabulary accepts nonempty training split ONLY")
    labels, attributes, predicates = set(), set(), set()
    for e in train:
        for graph in (e.source, e.target):
            labels.update(n.label for n in graph.nodes)
            attributes.update(a for n in graph.nodes for a in n.attributes)
            predicates.update(edge.predicate for edge in graph.edges)
    if not labels:
        raise ValueError("Training vocabulary has no node labels")
    return {
        "schema": "train-semantic-values-v3",
        "train_ids": [e.id for e in train],
        "labels": sorted(labels),
        "attributes": sorted(attributes) or ["visible"],
        "predicates": sorted(predicates) or ["related to"],
        "predicate_fallback_if_training_has_no_edges": not predicates,
        "attribute_fallback_if_training_has_no_attributes": not attributes,
        "copy_index": 0,
    }


def prototype_inputs(vocab):
    result = []
    for field in ("labels", "attributes", "predicates"):
        for i, value in enumerate(vocab[field]):
            # CLIP vectors for semantic values are derived solely from train.
            graph = EditGraph((Node("prototype", "object", (value,)),)) if field == "attributes" else EditGraph()
            if field == "labels":
                graph = EditGraph((Node("prototype", value),))
            result.append(PrototypeInput(f"train-vocab:{field}:{i}", graph, value if field == "predicates" else ""))
    return result


def prototype_vectors(store, vocab):
    tensors = {}
    inputs = prototype_inputs(vocab)
    for field in ("labels", "attributes", "predicates"):
        texts = []
        for item in inputs:
            if item.id.startswith(f"train-vocab:{field}:"):
                texts.append(
                    item.instruction if field == "predicates" else feature_texts(**item.model_input())["nodes"][0]
                )
        tensors[field] = torch.stack([store[t] for t in texts])
    return tensors


def field_index(vocab, field, value, before=None):
    if before is not None and value == before:
        return 0
    try:
        return vocab[field].index(value) + 1
    except ValueError:
        return -100  # OOV loss masked; example stays in ALL evaluation denominators.


def coverage(examples, vocab):
    missing, total_fields, oov_fields = [], 0, 0
    representable_ops, total_ops = 0, 0
    for e in examples:
        errors = []
        for op in e.operations:
            total_ops += 1
            local = []
            if op.after is not None:
                fields = (("labels", "label"),) if op.kind.endswith("node") else (("predicates", "predicate"),)
                for field, attr in fields:
                    total_fields += 1
                    value = getattr(op.after, attr)
                    old = getattr(op.before, attr) if op.before is not None else None
                    if field_index(vocab, field, value, old) == -100:
                        oov_fields += 1
                        local.append({"operation": op.kind, "entity": op.entity_id, "field": field, "value": value})
                if op.kind.endswith("node"):
                    old = set(op.before.attributes) if op.before else set()
                    for value in sorted(set(op.after.attributes) - old):
                        total_fields += 1
                        if field_index(vocab, "attributes", value) == -100:
                            oov_fields += 1
                            local.append(
                                {
                                    "operation": op.kind,
                                    "entity": op.entity_id,
                                    "field": "added_attribute",
                                    "value": value,
                                }
                            )
            representable_ops += not local
            errors.extend(local)
        if errors:
            missing.append({"id": e.id, "unrepresentable_fields": errors})
    return {
        "examples": len(examples),
        "fully_representable_examples": len(examples) - len(missing),
        "operations": total_ops,
        "representable_operation_payloads": representable_ops,
        "payload_fields": total_fields,
        "oov_payload_fields": oov_fields,
        "oov_cases": missing,
        "excluded_examples": 0,
        "scope": "payload vocabulary coverage, not a measured accuracy or full decoder upper bound",
    }
