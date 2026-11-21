"""One user invocation: cache -> six sequential runs -> heldout evaluation -> report."""
import argparse
import gc
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path

import torch

from src.datasets.gqa_full import atomic_write
from tools.edit_model_v1.model import EditPredictor
from tools.edit_model_v1.runner import source_hashes as original_source_hashes
from tools.edit_model_v1.runner import train_one, validate
from tools.emnlp2020_graph_modification_v1.dataset import fixed, read, sha256, write
from tools.prepare_magicbrush import ROOT, digest
from tools.run_gqa_full import exclusive

from .cache import build_cache
from .data import load_heldout, prepare_trainval
from .evaluation import freeze_selection, score_rows, seal_run, summarize, verify_receipt

DEFAULT_CONFIG = ROOT / "configs/level2/magicbrush-manual-experiment-v1.json"


def source_hashes():
    result = original_source_hashes()
    paths = [
        *Path(__file__).parent.glob("*.py"),
        *ROOT.glob("src/**/*.py"),
        ROOT / "tools/emnlp2020_graph_modification_v1/dataset.py",
        ROOT / "tools/run_gqa_full.py",
        ROOT / "scripts/run_magicbrush_manual_full602_v1.sh",
    ]
    result.update({str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)})
    package = Path(importlib.util.find_spec("clip").origin).parent
    result["installed_clip/bpe_simple_vocab_16e6.txt.gz"] = sha256(package / "bpe_simple_vocab_16e6.txt.gz")
    return result


def versions():
    return {k: importlib.metadata.version(k) for k in ("torch", "numpy", "clip")}


def validate_config(config, root):
    if config["schema"] != "magicbrush-manual-experiment-v1" or config["device"] != "cuda":
        raise ValueError("Expected versioned user-run CUDA experiment")
    if config["models"] != ["mlp", "gnn"] or config["seeds"] != [42, 123, 3407]:
        raise ValueError("Expected both models and the fixed three seeds")
    if set(config["artifacts"]) != {
        "original_manifest",
        "trainval_manifest",
        "trainval_edits",
        "test_manifest",
        "test_edits",
        "human_review",
        "reviewed_states",
    }:
        raise ValueError("Incomplete pinned artifact set")
    for key in ("text_batch_size", "batch_size", "val_batch_size", "epochs", "patience", "max_action_tokens"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError("Invalid budget: " + key)
    if (
        config["max_action_tokens"] != 2048
        or config["lr"] <= 0
        or config["weight_decay"] < 0
        or config["gradient_clip"] <= 0
    ):
        raise ValueError("Invalid optimization or codec budget")
    if (
        set(config["model"]) != {"hidden", "byte_dim", "max_nodes", "max_edges", "layers"}
        or any(type(v) is not int or v <= 0 for v in config["model"].values())
        or config["model"]["max_nodes"] != 20
        or config["model"]["max_edges"] != 60
    ):
        raise ValueError("Invalid model architecture")
    for key, parent in (("text_cache", "data"), ("test_text_cache", "data"), ("output", "runs")):
        if not (root / config[key]).resolve().is_relative_to((root / parent).resolve()):
            raise ValueError("Output outside allowed root: " + key)
    if config["text_cache"] == config["test_text_cache"]:
        raise ValueError("Development and test caches must be separate")


def preflight(config_path, *, root=ROOT):
    config = read(config_path)
    validate_config(config, root)
    lock_path = config_path.with_suffix(".lock.json")
    lock = read(lock_path)
    code = source_hashes()
    if lock != {"config_sha256": sha256(config_path), "source_sha256": code, "versions": versions()}:
        raise ValueError("Release lock differs from config/code/environment; use a new version, do not bypass the lock")
    if sha256(root / config["weights"]) != config["weights_sha256"]:
        raise ValueError("CLIP weights missing or changed")
    examples, report, paths = prepare_trainval(config, root)
    identity = {
        "schema": config["schema"],
        "config": config,
        "release_lock_sha256": sha256(lock_path),
        "source_sha256": code,
        "versions": versions(),
        "train_ids": [e.id for e in examples["train"]],
        "val_ids": [e.id for e in examples["val"]],
        "human_review": report["human_review"],
        "model_inputs": "source graph and frozen CLIP instruction/node/predicate embeddings only",
        "selection": "validation autoregressive micro operation F1; ties earlier epoch",
        "test_targets_used_for_training_or_selection": False,
    }
    report.update(
        stage="CPU preflight only",
        weights_sha256=config["weights_sha256"],
        experiment_sha256=digest(identity),
        code_and_environment_lock_verified=True,
        queue=[{"model": k, "seed": n} for k in config["models"] for n in config["seeds"]],
        automatic_queue=True,
        gpu_jobs_started=False,
        test_evaluation="all six selected checkpoints must be frozen before opening test labels",
    )
    return config, examples, report, identity, paths


def evaluate_one(selected, examples, store, config, experiment_sha, selection_sha, output, *, device="cuda"):
    directory = output / selected["model"] / str(selected["seed"])
    checkpoint = Path(selected["checkpoint"])
    if sha256(checkpoint) != selected["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint changed")
    result_path = directory / "test-result.json"
    predictions_path = directory / "predictions-test.json"
    expected = {
        "experiment_sha256": experiment_sha,
        "selection_sha256": selection_sha,
        "checkpoint_sha256": selected["checkpoint_sha256"],
        "model": selected["model"],
        "seed": selected["seed"],
        "test_cache_manifest_sha256": sha256(ROOT / config["test_text_cache"] / "manifest.json"),
    }
    if result_path.exists():
        result = read(result_path)
        if any(result.get(k) != v for k, v in expected.items()) or result["predictions_sha256"] != sha256(
            predictions_path
        ):
            raise ValueError("Test receipt changed")
        if score_rows(examples, read(predictions_path)) != result["metrics"]:
            raise ValueError("Test receipt metrics changed")
        return result
    # A crash after predictions were committed needs only CPU rescoring on resume.
    if predictions_path.exists():
        predictions = read(predictions_path)
    else:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["identity"] != {
            "experiment_sha256": experiment_sha,
            "model": selected["model"],
            "seed": selected["seed"],
        }:
            raise ValueError("Checkpoint payload identity mismatch")
        model = EditPredictor(kind=selected["model"], text_dim=store.dim, **config["model"]).to(device)
        model.load_state_dict(saved["model"])
        _, predictions = validate(model, examples, store, config, device)
        write(predictions_path, predictions)
        del model
    metrics = score_rows(examples, predictions)
    result = {
        **expected,
        "metrics": metrics,
        "predictions_sha256": sha256(predictions_path),
        "selected_epoch": selected["selected_epoch"],
        "selection_uses_test": False,
    }
    write(result_path, result)
    return result


def execute(config, examples, report, identity, paths, *, wandb_mode="offline"):
    output = ROOT / config["output"]

    def progress(stage, **extra):
        state = {"stage": stage, **extra}
        atomic_write(output / "progress.json", state)
        print(json.dumps(state, ensure_ascii=False), flush=True)

    with exclusive(ROOT / "runs/level2/.gqa-kg-training-device"), exclusive(output):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required: this stage is for the user to run")
        torch.set_num_threads(1)
        torch.backends.cudnn.benchmark = False
        write(output / "preflight.json", report)
        write(output / "base-experiment.json", identity)
        try:
            progress("cache_trainval")
            store, meta = build_cache(
                examples["train"] + examples["val"],
                config,
                identity,
                ROOT / config["text_cache"],
                ROOT,
                scope="trainval",
            )
            if store.dim != 512:
                raise ValueError("Expected frozen CLIP text dimension 512")
            training_identity = {
                **identity,
                "text_cache_manifest_sha256": sha256(ROOT / config["text_cache"] / "manifest.json"),
                "text_vectors_sha256": meta["vectors_sha256"],
            }
            experiment_sha = digest(training_identity)
            write(output / "experiment.json", training_identity)
            for kind in config["models"]:
                for seed in config["seeds"]:
                    directory = output / kind / str(seed)
                    progress("train", model=kind, seed=seed)
                    if (directory / "completed-receipt.json").exists():
                        verify_receipt(directory, experiment_sha, kind, seed)
                    else:
                        train_one(
                            examples, store, config, training_identity, kind, seed, directory, wandb_mode=wandb_mode
                        )
                        seal_run(directory, experiment_sha, kind, seed, examples["val"])
                    gc.collect()
                    torch.cuda.empty_cache()
            selection = freeze_selection(output, config, experiment_sha)
            selection_sha = sha256(output / "selection.json")
            progress("load_heldout_after_all_training")
            test, coverage = load_heldout(paths, config["expected_counts"]["test"], config["max_action_tokens"])
            test_identity = {
                "base_experiment_sha256": digest(identity),
                "selection_sha256": selection_sha,
                "scope": "test source inputs only after all checkpoint selection",
                "input_ids": [e.id for e in test],
            }
            progress("cache_test_sources")
            test_store, _ = build_cache(
                test, config, test_identity, ROOT / config["test_text_cache"], ROOT, scope="test_after_selection"
            )
            for selected in selection["checkpoints"]:
                progress("evaluate_test", model=selected["model"], seed=selected["seed"])
                evaluate_one(selected, test, test_store, config, experiment_sha, selection_sha, output)
                gc.collect()
                torch.cuda.empty_cache()
            progress("summarize")
            summary = summarize(output, config, experiment_sha, examples["val"], test, coverage)
            lines = [
                "MagicBrush full602 experiment completed",
                "Metrics are against the frozen accepted AI-assisted labels; manual review covered 20 train cases.",
                f'Test coverage: {coverage["evaluated_pairs"]}/{coverage["original_test_pairs"]}; deferred: {coverage["deferred_pairs"]}.',
            ]
            for kind, splits in summary["models"].items():
                for split, result in splits.items():
                    metric = result["aggregate"]["micro"]["f1"]
                    lines.append(f'{kind} {split} micro F1: mean={metric["mean"]}, sample std={metric["std"]}')
            lines.append(
                "Full per-operation PRF1, exact match, failure rates, strata and no-edit baseline: summary.json"
            )
            fixed(output / "SUMMARY.txt", ("\n".join(lines) + "\n").encode())
            progress("complete", summary=str(output / "summary.json"))
        except BaseException as error:
            progress(
                "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                error=str(error),
                resume="Rerun the same command; completed runs are verified and skipped, unfinished runs resume at saved epoch.",
            )
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=("plan", "run"), default="plan")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="offline")
    parser.add_argument("--report", type=Path, help="Optional CPU plan report destination")
    args = parser.parse_args(argv)
    config, examples, report, identity, paths = preflight(args.config.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.report:
        write(args.report, report)
    if args.stage == "plan":
        return 0
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    execute(config, examples, report, identity, paths, wandb_mode=args.wandb_mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
