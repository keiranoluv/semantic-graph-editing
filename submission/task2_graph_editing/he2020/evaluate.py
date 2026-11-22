"""Evaluate one He graph-edit checkpoint without a historical audit queue."""

import argparse
import json
from pathlib import Path

import torch
from common.editing.text import TextStore
from common.io import read, sha256
from task2_graph_editing.he2020.dataset import load_bundle
from task2_graph_editing.he2020.predict import load_model
from task2_graph_editing.he2020.runtime import evaluate, inputs_tensor, predict


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model", choices=("mlp", "gnn"), required=True)
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--max-examples", type=int)
    a = p.parse_args()
    if a.max_examples is not None and a.max_examples < 1:
        p.error("--max-examples must be positive")
    model, vocab, config = load_model(a.checkpoint, a.model)
    dataset = Path(config["dataset"])
    examples = load_bundle(dataset, "test" if a.split == "test" else "trainval")[a.split]
    if a.max_examples is not None:
        examples = examples[: a.max_examples]
    cache = dataset / ("clip-test" if a.split == "test" else "clip-trainval")
    metadata = read(cache / "manifest.json")
    if sha256(cache / "vectors.pt") != metadata["vectors_sha256"]:
        raise ValueError("Frozen text feature cache changed")
    store = TextStore(metadata["text_keys"], torch.load(cache / "vectors.pt", map_location="cpu", weights_only=True))
    tensors = inputs_tensor(examples, store, config)
    predictions = predict(model, tensors, vocab, config)
    metrics, _, _ = evaluate(examples, predictions, vocab)
    print(json.dumps({"examples": len(examples), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
