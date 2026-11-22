"""Predict graph edits from one source graph and instruction on CPU."""

import argparse
import json
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import torch
from common.editing.operations import EditGraph, apply_operations
from common.io import read, sha256
from task2_graph_editing.he2020.data_processing.model import ActionVocabulary, frozen_features
from task2_graph_editing.he2020.runtime import make_model, predict
from task2_graph_editing.magicbrush.sequence_model.codec import canonical_actions, canonical_source, decode_actions
from task2_graph_editing.magicbrush.sequence_model.model import collate_inputs
from task2_graph_editing.magicbrush.structured_model.vocabulary import PrototypeInput


def load_model(checkpoint, kind):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = {**saved["config"], "device": "cpu"}
    vocab = ActionVocabulary(saved["vocabulary"])
    model = make_model(config, kind, vocab)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model, vocab, config


def infer(checkpoint, kind, item, cache):
    source, node_ids, edge_ids = canonical_source(EditGraph.from_dict(item["source_graph"]))
    example = PrototypeInput("inference", source, item["instruction"])
    model, vocab, config = load_model(checkpoint, kind)
    inputs = [example.model_input()]
    with redirect_stdout(sys.stderr):
        store = frozen_features(inputs, Path(cache), config["weights"], config["text_batch_size"])
    batch = collate_inputs(inputs, store, max_nodes=config["max_nodes"], max_edges=config["max_edges"], device="cpu")
    tokens = predict(model, batch, vocab, config)[0]
    result = {
        "checkpoint_sha256": sha256(checkpoint),
        "source_graph": source.to_dict(),
        "input_node_ids": node_ids,
        "input_edge_ids": edge_ids,
        "operations": None,
        "target_graph": None,
        "error": None,
    }
    try:
        operations = canonical_actions(source, decode_actions(tokens, source))
        result.update(
            operations=[o.to_dict() for o in operations], target_graph=apply_operations(source, operations).to_dict()
        )
    except (ValueError, TypeError, KeyError, UnicodeError) as exc:
        result["error"] = str(exc)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model", choices=("mlp", "gnn"), required=True)
    p.add_argument("--input", type=Path, required=True)
    a = p.parse_args()
    with tempfile.TemporaryDirectory(prefix="he-inference-") as cache:
        print(json.dumps(infer(a.checkpoint, a.model, read(a.input), cache), indent=2))


if __name__ == "__main__":
    main()
