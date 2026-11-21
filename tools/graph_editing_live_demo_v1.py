"""CPU inference for the submission notebook using the selected MagicBrush GNN."""

import torch

from experiments.magicbrush_prompt_intent_v1.dataset import MANIFEST
from experiments.magicbrush_prompt_intent_v1.evaluate import REPORT
from experiments.magicbrush_prompt_intent_v1.run import checked_json, text_store
from src.editing.operations import apply_operations
from tools.edit_model_v1.codec import decode_actions
from tools.edit_model_v1.model import collate_inputs
from tools.emnlp2020_graph_modification_v1.dataset import read, sha256
from tools.magicbrush_experiment_v3.decoding import decode_batch
from tools.magicbrush_experiment_v3.model import StructuredPredictor
from tools.magicbrush_experiment_v3.vocabulary import PrototypeInput, prototype_vectors
from tools.prepare_magicbrush import ROOT, digest


def predict(source, instruction):
    """Return decoded operations, the resulting graph and checkpoint provenance."""
    selection = checked_json(REPORT / "selection.json", read(REPORT / "summary.json")["selection_sha256"])
    entry = next(row for row in selection if row["policy"] == "prompt" and row["seed"] == 42)
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
