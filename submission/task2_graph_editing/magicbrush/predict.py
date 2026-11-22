"""Load a graph-edit checkpoint and predict operations from a source graph."""

from pathlib import Path

import torch
from common.editing.operations import apply_operations
from common.io import read, sha256
from common.paths import ROOT
from task2_graph_editing.magicbrush.dataset import MANIFEST
from task2_graph_editing.magicbrush.examples import checked_json
from task2_graph_editing.magicbrush.features import text_store
from task2_graph_editing.magicbrush.preprocess import digest
from task2_graph_editing.magicbrush.sequence_model.codec import decode_actions
from task2_graph_editing.magicbrush.sequence_model.model import collate_inputs
from task2_graph_editing.magicbrush.structured_model.decoding import decode_batch
from task2_graph_editing.magicbrush.structured_model.model import StructuredPredictor
from task2_graph_editing.magicbrush.structured_model.vocabulary import PrototypeInput, prototype_vectors


def load_model(examples, *, seed=42, policy="prompt", checkpoint=None):
    entry = None
    if checkpoint is None:
        entry = next(
            r for r in read(ROOT / "assets/magicbrush_checkpoints.json") if r["policy"] == policy and r["seed"] == seed
        )
        path = ROOT / entry["checkpoint"]
        if sha256(path) != entry["sha256"]:
            raise ValueError("Checkpoint checksum mismatch")
    else:
        path = ROOT / Path(checkpoint)
    protocol = read(path.parent / "protocol.json")
    best = torch.load(path, map_location="cpu", weights_only=True)
    if digest(protocol) != best["identity"]["experiment_sha256"] or protocol["train_policy"] != policy:
        raise ValueError("Checkpoint and protocol differ")
    if entry is not None and (best["identity"] != entry["identity"] or best["epoch"] != entry["selected_epoch"]):
        raise ValueError("Checkpoint differs from the selected experiment")
    manifest = checked_json(MANIFEST, protocol["dataset_manifest_sha256"])
    config = checked_json(ROOT / manifest["parent_config"], manifest["parent_config_sha256"])
    for key in ("weights", "text_cache"):
        config[key] = str(ROOT / config[key])
    for example in examples:
        if (
            len(example.source.nodes) > config["model"]["max_nodes"]
            or len(example.source.edges) > config["model"]["max_edges"]
        ):
            raise ValueError("Source graph exceeds checkpoint capacity")
    vocab = protocol["vocabulary"]
    store, _ = text_store(config, manifest, examples, vocab)
    model = StructuredPredictor(
        kind="gnn", vocabulary=vocab, prototypes=prototype_vectors(store, vocab), **config["model"]
    )
    model.load_state_dict(best["model"])
    model.eval()
    if entry is None:
        entry = {
            "checkpoint": str(path),
            "sha256": sha256(path),
            "selected_epoch": best["epoch"],
            "identity": best["identity"],
            "seed": seed,
            "policy": policy,
        }
    return model, store, config, entry


def predict(source, instruction, *, seed=42):
    example = PrototypeInput("live-demo", source, instruction)
    model, store, config, entry = load_model([example], seed=seed)
    items = [example.model_input()]
    with torch.inference_mode():
        batch = collate_inputs(items, store, device="cpu")
        (row,) = decode_batch(items, model(batch), model.vocabulary, model.max_nodes)
    operations = decode_actions(row["tokens"], source)
    return operations, apply_operations(source, operations), entry
