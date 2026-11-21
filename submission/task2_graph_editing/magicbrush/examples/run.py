"""Run the paired, fixed-budget CPU pilot from the repository root."""

import argparse
import hashlib
import json
import random
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from common.editing.operations import EditGraph, Operation, apply_operations
from task2_graph_editing.magicbrush.sequence_model.codec import canonical_actions
from task2_graph_editing.magicbrush.sequence_model.data import Example
from task2_graph_editing.magicbrush.sequence_model.text import load_store
from common.io import read, sha256, write

from task2_graph_editing.magicbrush.structured_model.model import StructuredPredictor
from task2_graph_editing.magicbrush.structured_model.runtime import training_step, validate
from task2_graph_editing.magicbrush.structured_model.vocabulary import build_vocabulary, prototype_vectors

PROTOCOL = Path(__file__).with_name("protocol.json")
DEFAULT_OUTPUT = Path("runs/level2/magicbrush-prompt-pilot-v1")


def checked_json(path, expected):
    if sha256(path) != expected:
        raise ValueError(f"Changed input: {path}")
    return read(path)


def example(row, ops):
    source = EditGraph.from_dict(row["source_graph"])
    actions = canonical_actions(source, [Operation.from_dict(op) for op in ops])
    return Example(
        row["id"],
        row["split"],
        row["source_group_id"],
        row["instruction"],
        source,
        apply_operations(source, actions),
        actions,
    )


def inputs(protocol, config):
    files = config["artifacts"]
    rows = {
        r["id"]: r for r in checked_json(files["trainval_edits"]["path"], files["trainval_edits"]["sha256"])["records"]
    }
    manifest = checked_json(files["trainval_manifest"]["path"], files["trainval_manifest"]["sha256"])
    media = {r["id"]: r for r in manifest["records"]}
    cohort = checked_json(protocol["cohort"], protocol["cohort_sha256"])["cases"]
    assert [r["id"] for r in protocol["records"]] == [r["id"] for r in cohort]
    train = {"graph_diff": [], "prompt": []}
    for decision in protocol["records"]:
        r = rows[decision["id"]]
        assert r["split"] == "train" and decision["prompt"] == r["instruction"]
        for role, digest in decision["images"].items():
            assert media[r["id"]][role]["sha256"] == digest
        if decision["operations"] is None:
            continue
        old = example(r, r["operations"])
        assert old.target == EditGraph.from_dict(r["target_graph"])
        train["graph_diff"].append(old)
        train["prompt"].append(example(r, decision["operations"]))

    proposal = checked_json(protocol["validation"], protocol["validation_sha256"])
    validation, excluded = [], []
    train_ids = {e.id for e in train["prompt"]}
    train_groups = {e.group for e in train["prompt"]}
    train_pixels = {media[i][k]["pixel_sha256"] for i in train_ids for k in ("source_image", "target_image")}
    for decision in proposal["records"]:
        r = rows[decision["id"]]
        assert r["split"] == "val" and r["source_group_id"] not in train_groups
        assert r["instruction"] == decision["prompt"]
        assert all(media[r["id"]][k]["pixel_sha256"] not in train_pixels for k in ("source_image", "target_image"))
        if decision["complete_reference_proposed"] and not decision["alternative_operations"]:
            validation.append(example(r, decision["candidate_operations"]))
        else:
            excluded.append(r["id"])
    assert len(train_ids) == 14 and len(validation) == 13 and len(excluded) == 7
    return train, validation, excluded


def main(output):
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    protocol = read(PROTOCOL)
    config = checked_json(protocol["parent"], protocol["parent_sha256"])
    train, validation, excluded = inputs(protocol, config)
    cache = Path(config["text_cache"])
    checked_json(cache / "manifest.json", protocol["cache_manifest_sha256"])
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    store, _ = load_store(cache, read(cache / "identity.json"))
    # Both heads have identical classes and initialization. All prototype values
    # come from the two annotations of the same training IDs.
    vocab = build_vocabulary(train["graph_diff"] + train["prompt"])
    vectors = prototype_vectors(store, vocab)
    seed, batch = protocol["seed"], protocol["batch_size"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temp:
        stage = Path(temp) / "results"
        stage.mkdir()
        summary = {
            "protocol_sha256": sha256(PROTOCOL),
            "runner_sha256": sha256(__file__),
            "torch": str(torch.__version__),
            "seed": seed,
            "batch": batch,
            "epochs": protocol["epochs"],
            "validation_ids": [e.id for e in validation],
            "excluded_validation_ids": excluded,
            "train_ids": [e.id for e in train["prompt"]],
            "runs": {},
        }
        for policy, examples in train.items():
            torch.manual_seed(seed)
            rng = random.Random(seed)
            model = StructuredPredictor(kind="gnn", vocabulary=vocab, prototypes=vectors, **config["model"])
            init = hashlib.sha256()
            for tensor in model.state_dict().values():
                init.update(tensor.detach().contiguous().numpy().tobytes())
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
            losses = []
            for _ in range(protocol["epochs"]):
                order = rng.sample(examples, len(examples))
                for start in range(0, len(order), batch):
                    loss, _ = training_step(model, optimizer, order[start : start + batch], store, config, "cpu")
                    losses.append(loss)
            fit, _ = validate(model, examples, store, config, "cpu")
            val, predictions = validate(model, validation, store, config, "cpu")
            torch.save({"vocabulary": vocab, "model": model.state_dict()}, stage / f"{policy}.pt")
            write(stage / f"{policy}-predictions.json", predictions)
            summary["runs"][policy] = {
                "initial_state_sha256": init.hexdigest(),
                "steps": len(losses),
                "first_loss": losses[0],
                "last_loss": losses[-1],
                "train": fit,
                "validation": val,
                "checkpoint_sha256": sha256(stage / f"{policy}.pt"),
            }
        assert len({v["initial_state_sha256"] for v in summary["runs"].values()}) == 1
        write(stage / "summary.json", summary)
        stage.rename(output)
    print(json.dumps({k: v["validation"]["micro"] for k, v in summary["runs"].items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    main(parser.parse_args().output)
