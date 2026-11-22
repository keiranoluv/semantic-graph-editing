"""Construct MLP/GNN models, prepare batches, decode operations and compute edit metrics."""

from collections import Counter

import torch
from common.editing.codec import canonical_actions, decode_actions, score_predictions
from common.editing.encoders import collate_inputs
from common.io import digest
from task2_graph_editing.he2020.model import LexicalPredictor


def identity(config):
    from common.provenance import source_hashes

    return {"config": config, "files": source_hashes()}


def tensor_subset(tensors, indices, device):
    return {key: value[indices].to(device) for key, value in tensors.items()}


def predict(model, tensors, vocabulary, config):
    predictions = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(tensors["instruction"]), config["batch_size"]):
            batch = {k: v[start : start + config["batch_size"]].to(config["device"]) for k, v in tensors.items()}
            predictions.extend(vocabulary.to_bytes(row) for row in model.predict(batch, config["max_decode_tokens"]))
    return predictions


def evaluate(examples, predictions, vocabulary):
    metrics, errors = score_predictions(examples, predictions)
    oov = [3 in vocabulary.encode(e.operations) for e in examples]
    metrics["reference_programs_with_train_vocabulary_OOV"] = sum(oov)
    metrics["closed_vocabulary_exact_match_upper_bound"] = 1 - sum(oov) / len(oov) if oov else None
    metrics["by_reference_edit_kind"] = {}
    for kind in ("add_node", "delete_node", "replace_node", "replace_edge"):
        indices = [i for i, e in enumerate(examples) if any(o.kind == kind for o in e.operations)]
        if indices:
            metrics["by_reference_edit_kind"][kind] = score_predictions(
                [examples[i] for i in indices], [predictions[i] for i in indices]
            )[0]
    cases = []
    for e, tokens, error, unknown in zip(examples, predictions, errors, oov, strict=True):
        ops = ()
        try:
            ops = decode_actions(tokens, e.source)
            ops = canonical_actions(e.source, ops)
        except (ValueError, TypeError, KeyError, UnicodeError):
            pass
        exact = error is None and set(ops) == set(e.operations)
        cases.append(
            {
                "id": e.id,
                "correct": exact,
                "error": error,
                "reference_OOV": unknown,
                "instruction": e.instruction,
                "source": e.source.to_dict(),
                "target": e.target.to_dict(),
                "expected": [o.to_dict() for o in e.operations],
                "predicted": [o.to_dict() for o in ops],
            }
        )
    metrics["error_categories"] = dict(
        Counter("correct" if c["correct"] else (c["error"] or "valid_but_wrong").split(":")[0] for c in cases)
    )
    selected = []
    for correct in (True, False):
        selected += sorted((c for c in cases if c["correct"] == correct), key=lambda c: digest([42, c["id"]]))[:5]
    return metrics, cases, selected


def make_model(config, kind, vocabulary):
    return LexicalPredictor(
        kind=kind,
        vocabulary_size=vocabulary.size,
        text_dim=512,
        hidden=config["hidden"],
        byte_dim=48,
        layers=config["layers"],
        max_nodes=config["max_nodes"],
        max_edges=config["max_edges"],
    ).to(config["device"])


def inputs_tensor(examples, store, config):
    if any(len(e.source.nodes) > config["max_nodes"] or len(e.source.edges) > config["max_edges"] for e in examples):
        raise ValueError("source_identity_budget_exceeded")
    return collate_inputs(
        [e.model_input() for e in examples],
        store,
        max_nodes=max(1, max(len(e.source.nodes) for e in examples)),
        max_edges=max(1, max(len(e.source.edges) for e in examples)),
        device="cpu",
    )
