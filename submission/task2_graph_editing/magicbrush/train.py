"""Paired GNN training on MagicBrush prompt labels; user-run CUDA, validation only."""

import argparse
import importlib.metadata
import os
from pathlib import Path

import torch
from common.io import read, sha256, write
from common.locking import exclusive
from common.paths import ROOT
from common.provenance import source_hashes
from task2_graph_editing.magicbrush.dataset import HERE, MANIFEST, rows
from task2_graph_editing.magicbrush.examples import checked_json, example
from task2_graph_editing.magicbrush.features import text_store
from task2_graph_editing.magicbrush.model import StructuredPredictor
from task2_graph_editing.magicbrush.preprocess import digest
from task2_graph_editing.magicbrush.runtime import training_step, validate
from task2_graph_editing.magicbrush.targets import encode_targets
from task2_graph_editing.magicbrush.training import train_one
from task2_graph_editing.magicbrush.vocabulary import build_vocabulary, prototype_vectors

OUTPUT = ROOT / "runs/level2/magicbrush-prompt-intent-v1"


def prepare():
    manifest = read(MANIFEST)
    config = checked_json(manifest["parent_config"], manifest["parent_config_sha256"])
    lock = read(Path(manifest["parent_config"]).with_suffix(".lock.json"))
    if lock["config_sha256"] != manifest["parent_config_sha256"]:
        raise ValueError("Frozen parent implementation differs")
    source_hashes()
    train = rows("train")
    arms = {
        name: [example(r, r[key]) for r in train]
        for name, key in [("graph_diff", "graph_diff_operations"), ("prompt", "operations")]
    }
    validation = [example(r, r["operations"]) for r in rows("val")]
    vocab = build_vocabulary(arms["graph_diff"] + arms["prompt"])
    vocab["train_ids"] = [r["id"] for r in train]
    store, delta = text_store(config, manifest, arms["prompt"] + validation, vocab)
    for partition in [*arms.values(), validation]:
        for start in range(0, len(partition), config["batch_size"]):
            encode_targets(partition[start : start + config["batch_size"]], vocab, config, "cpu")
    identity = {
        "experiment": "magicbrush-prompt-intent-v1",
        "dataset_manifest_sha256": sha256(MANIFEST),
        "parent_lock_sha256": digest(lock),
        "runtime_versions": versions(),
        "runner_sha256": sha256(__file__),
        "dataset_reader_sha256": sha256(HERE / "dataset.py"),
        "example_builder_sha256": sha256(HERE / "examples.py"),
        "cache_manifest_sha256": manifest["cache_manifest_sha256"],
        "clip_cpu_extension": delta,
        "vocabulary": vocab,
        "validation_ids": [e.id for e in validation],
        "validation_policy": "prompt",
    }
    return (arms, validation, config, store, identity)


def smoke(arms, validation, config, store, vocab):
    prototypes = prototype_vectors(store, vocab)
    for name, train in arms.items():
        torch.manual_seed(config["seeds"][0])
        model = StructuredPredictor(kind="gnn", vocabulary=vocab, prototypes=prototypes, **config["model"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        loss, size = training_step(model, optimizer, train[: config["batch_size"]], store, config, "cpu")
        _, predictions = validate(model, validation[:2], store, config, "cpu")
        print(
            f"CPU {name}: {len(train)} train, {len(validation)} val, batch={size}, loss={loss:.4f}, decoded={len(predictions)}",
            flush=True,
        )


def main(cpu_check):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    arms, val, config, store, identity = prepare()
    if cpu_check:
        smoke(arms, val, config, store, identity["vocabulary"])
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for the training queue")
    os.environ["WANDB_RUN_GROUP"] = "magicbrush-prompt-intent-v1"
    with exclusive(ROOT / "runs/level2/.gqa-kg-training-device"), exclusive(OUTPUT):
        for seed in config["seeds"]:
            for policy, train in arms.items():
                directory = OUTPUT / policy / f"seed-{seed}"
                os.environ["WANDB_JOB_TYPE"] = policy
                run_identity = {**identity, "train_policy": policy}
                write(directory / "protocol.json", run_identity)
                train_one(
                    {"train": train, "val": val},
                    store,
                    config,
                    run_identity,
                    "gnn",
                    seed,
                    directory,
                    device="cuda",
                    wandb_mode="online",
                )


def versions():
    return {k: importlib.metadata.version(k) for k in ("torch", "numpy", "clip")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-check", action="store_true")
    main(parser.parse_args().cpu_check)
