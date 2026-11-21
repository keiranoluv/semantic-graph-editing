"""CPU inference for the submission notebook using the selected MagicBrush GNN."""

import torch

from task2_graph_editing.magicbrush.dataset import MANIFEST
from task2_graph_editing.magicbrush.evaluate import REPORT
from task2_graph_editing.magicbrush.run import checked_json, text_store
from common.editing.operations import apply_operations
from task2_graph_editing.magicbrush.sequence_model.codec import decode_actions
from task2_graph_editing.magicbrush.sequence_model.model import collate_inputs
from common.io import read, sha256

from task2_graph_editing.magicbrush.structured_model.decoding import decode_batch
from task2_graph_editing.magicbrush.structured_model.model import StructuredPredictor
from task2_graph_editing.magicbrush.structured_model.vocabulary import PrototypeInput, prototype_vectors
from task2_graph_editing.magicbrush.preprocess import ROOT, digest


def predict(source, instruction, *, seed=42):
    """Return decoded operations, the resulting graph and checkpoint provenance."""
    selection = checked_json(REPORT / "selection.json", read(REPORT / "summary.json")["selection_sha256"])
    entry = next(row for row in selection if row["policy"] == "prompt" and row["seed"] == seed)
    path = ROOT / entry["checkpoint"]
    if sha256(path) != entry["sha256"]:
        raise ValueError("Checkpoint checksum mismatch")
    protocol = read(path.parent / "protocol.json")
    best = torch.load(path, map_location="cpu", weights_only=True)
    if (
        best["identity"] != entry["identity"]
        or best["epoch"] != entry["selected_epoch"]
        or digest(protocol) != entry["identity"]["experiment_sha256"]
    ):
        raise ValueError("Checkpoint differs from the selected experiment")
    manifest = checked_json(MANIFEST, protocol["dataset_manifest_sha256"])
    config = checked_json(ROOT / manifest["parent_config"], manifest["parent_config_sha256"])
    for key in ("weights", "text_cache"):
        config[key] = str(ROOT / config[key])
    for field, values in (("max_nodes", source.nodes), ("max_edges", source.edges)):
        if len(values) > config["model"][field]:
            raise ValueError(f"Graph exceeds checkpoint capacity: {field}={config['model'][field]}")
    example = PrototypeInput("live-demo", source, instruction)
    vocab = protocol["vocabulary"]
    store, _ = text_store(config, manifest, [example], vocab)
    model = StructuredPredictor(
        kind="gnn", vocabulary=vocab, prototypes=prototype_vectors(store, vocab), **config["model"]
    )
    model.load_state_dict(best["model"])
    model.eval()
    items = [example.model_input()]
    with torch.inference_mode():
        inputs = collate_inputs(items, store, device="cpu")
        (row,) = decode_batch(items, model(inputs), vocab, model.max_nodes)
    operations = decode_actions(row["tokens"], source)
    return operations, apply_operations(source, operations), entry
