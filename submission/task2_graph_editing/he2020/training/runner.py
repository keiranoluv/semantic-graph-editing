"""Preflight/smoke without W&B, user-launched main queue with online W&B."""

import argparse
import copy
import importlib.metadata
import json
import math
import os
import subprocess
from pathlib import Path

import torch
from common.io import digest, read, sha256, write
from common.paths import ROOT
from task2_graph_editing.he2020.data_processing.data import load_bundle
from task2_graph_editing.he2020.data_processing.model import WEIGHTS_SHA256, ActionVocabulary
from task2_graph_editing.he2020.runtime import identity as base_identity
from task2_graph_editing.he2020.runtime import inputs_tensor
from task2_graph_editing.he2020.training.training import train_one
from task2_graph_editing.magicbrush.sequence_model.text import TextStore


def validate_config(config):
    for key in ("epochs", "minimum_epochs", "patience", "batch_size", "text_batch_size", "threads"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError("invalid_positive_integer:" + key)
    if (
        config["minimum_epochs"] > config["epochs"]
        or not math.isfinite(config["min_delta"])
        or config["min_delta"] < 0
        or config.get("validation_interval", 1) != 1
    ):
        raise ValueError("invalid_early_stopping_configuration")
    for key in ("lr", "gradient_clip"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError("invalid_positive_scalar:" + key)
    if not math.isfinite(config["weight_decay"]) or config["weight_decay"] < 0:
        raise ValueError("invalid_weight_decay")
    if not config["models"] or set(config["models"]) - {"mlp", "gnn"}:
        raise ValueError("invalid_models")
    if len(set(config["models"])) != len(config["models"]) or len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("duplicate_model_or_seed")
    if not config["seeds"] or config["device"] not in ("cpu", "cuda"):
        raise ValueError("invalid_seed_or_device")
    if config["execution"] == "main":
        if config["wandb"]["mode"] != "online" or os.environ.get("WANDB_MODE", "online") != "online":
            raise ValueError("main_training_requires_online_WandB")
        if config["device"] != "cuda":
            raise ValueError("main_GPU_queue_requires_CUDA_use_smoke_for_CPU_tests")
    elif config["execution"] != "smoke":
        raise ValueError("unknown_execution_scope")
    output = Path(config["output"]).resolve()
    if output in {
        Path(config["dataset"]).resolve(),
        (ROOT / "runs/level2/emnlp2020-task-b-gpu-v3").resolve(),
        (ROOT / "runs/level2/emnlp2020-task-b-v2").resolve(),
    }:
        raise ValueError("v4_requires_new_output_preserve_frozen_runs")


def identity(config):
    result = base_identity(config)
    result["files"].update({str(p.relative_to(ROOT)): sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))})
    return result


def load_features(config):
    directory = Path(config["dataset"]) / "clip-trainval"
    meta = read(directory / "manifest.json")
    if (
        meta["identity"]["weights_sha256"] != WEIGHTS_SHA256
        or sha256(directory / "vectors.pt") != meta["vectors_sha256"]
        or sha256(config["weights"]) != WEIGHTS_SHA256
    ):
        raise ValueError("frozen_CLIP_features_or_weights_changed")
    vectors = torch.load(directory / "vectors.pt", weights_only=True, map_location="cpu")
    if meta["shape"] != list(vectors.shape) or meta["target_text_encoded"]:
        raise ValueError("invalid_source_only_feature_metadata")
    return TextStore(meta["text_keys"], vectors)


def prepare_inputs(config):
    examples = load_bundle(config["dataset"], "trainval")
    if not examples["train"] or not examples["val"]:
        raise ValueError("empty_train_or_validation")
    if {e.group for e in examples["train"]} & {e.group for e in examples["val"]}:
        raise ValueError("source_graph_leakage")
    vocabulary = ActionVocabulary.fit(examples["train"])
    store = load_features(config)
    tensors = {split: inputs_tensor(rows, store, config) for split, rows in examples.items()}
    return examples, vocabulary, store, tensors


def experiment(config, vocabulary):
    directory = Path(config["output"])
    source = identity(config)
    inputs = {
        "trainval": sha256(Path(config["dataset"]) / "trainval-receipt.json"),
        "features": sha256(Path(config["dataset"]) / "clip-trainval/manifest.json"),
    }
    identifier = digest([source, inputs, vocabulary.tokens])
    path = directory / "experiment.json"
    if path.exists():
        old = read(path)
        if old["identity"] != identifier or old["source"] != source or old["inputs"] != inputs:
            raise ValueError("experiment_code_config_or_inputs_changed_use_new_version")
        # Keep creation commit stable when later commits contain only reports.
        return old
    result = {
        "identity": identifier,
        "source": source,
        "inputs": inputs,
        "creation_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "vocabulary": vocabulary.tokens,
        "test_used": False,
        "torch": str(torch.__version__),
        "wandb_version": importlib.metadata.version("wandb"),
        "test_was_already_evaluated_by_v3": True,
        "scope": "new_training_recipe_selected_from_v3_validation_histories_only",
    }
    write(path, result)
    return result


def train(config):
    validate_config(config)
    examples, vocabulary, _, tensors = prepare_inputs(config)
    info = experiment(config, vocabulary)
    runs = []
    for kind in config["models"]:
        for seed in config["seeds"]:
            result = train_one(config, kind, seed, examples, tensors, vocabulary, info["identity"])
            path = Path(config["output"]) / f"{kind}-seed{seed}"
            runs.append(
                {
                    "model": kind,
                    "seed": seed,
                    "checkpoint_sha256": result["checkpoint_sha256"],
                    "result_sha256": sha256(path / "result.json"),
                    "completed_receipt_sha256": sha256(path / "completed-receipt.json"),
                    "wandb": result["wandb"],
                }
            )
    freeze = {
        "experiment_identity": info["identity"],
        "runs": runs,
        "test_used": False,
        "automatic_test": False,
        "all_runs_tracked_online": all(r["wandb"] is not None for r in runs),
    }
    write(Path(config["output"]) / "frozen-checkpoints.json", freeze)
    return freeze


def preflight(config):
    validate_config(config)
    examples, vocabulary, _, _ = prepare_inputs(config)
    return {
        "examples": {k: len(v) for k, v in examples.items()},
        "vocabulary_size": vocabulary.size,
        "code_and_config_sha256": digest(identity(config)),
        "device_workloads": 0,
        "wandb_initialized": False,
        "main_run_not_started": True,
        "queue": [f"{kind}/seed{seed}" for kind in config["models"] for seed in config["seeds"]],
        "wandb_required": config["wandb"]["mode"],
        "authentication": "verified_by_online_init_before_first_gradient",
    }


def smoke(config, device):
    examples, vocabulary, store, _ = prepare_inputs(config)
    original = examples["train"]
    train_examples = original[:8]
    groups = {e.group for e in train_examples}
    validation = [e for e in original[8:] if e.group not in groups][:4]
    mini = {"train": train_examples, "val": validation}
    config = copy.deepcopy(config)
    config.update(
        execution="smoke",
        device=device,
        output=config["output"] + "-smoke-" + device,
        epochs=1,
        minimum_epochs=1,
        patience=1,
        seeds=[42],
    )
    validate_config(config)
    tensors = {split: inputs_tensor(rows, store, config) for split, rows in mini.items()}
    run_id = digest([identity(config), [e.id for rows in mini.values() for e in rows]])
    results = [train_one(config, kind, 42, mini, tensors, vocabulary, run_id) for kind in config["models"]]
    return {
        "execution": "smoke",
        "device": device,
        "wandb_initialized": False,
        "models": [r["model"] for r in results],
        "train_examples": len(train_examples),
        "diagnostic_val_from_official_train": len(validation),
        "epochs": 1,
        "seed": 42,
        "batch_size": config["batch_size"],
        "output": config["output"],
        "not_a_main_experiment_or_benchmark": True,
    }


def main():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/level2/emnlp2020-task-b-wandb-v4.json"))
    parser.add_argument("--stage", required=True, choices=("preflight", "smoke", "train"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="Smoke device only")
    args = parser.parse_args()
    config = read(args.config)
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.stage != "preflight" and (args.stage == "train" or args.device == "cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA_unavailable_no_automatic_CPU_fallback")
    if args.stage == "preflight":
        result = preflight(config)
    elif args.stage == "smoke":
        result = smoke(config, args.device)
    else:
        result = train(config)
    print(json.dumps(result, indent=2))
