"""One user invocation: cache -> six sequential runs -> heldout evaluation -> report."""
import argparse
import gc
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path

import torch

from common.datasets.gqa_full import atomic_write
from task2_graph_editing.magicbrush.sequence_model.runner import source_hashes as original_source_hashes
from common.io import fixed, read, sha256, write

from task2_graph_editing.magicbrush.data_processing.data import load_heldout, prepare_trainval
from task2_graph_editing.magicbrush.data_processing.evaluation import score_rows, seal_run, verify_receipt
from task2_graph_editing.magicbrush.preprocess import ROOT, digest
from task2_gqa.run_gqa_full import exclusive

from task2_graph_editing.magicbrush.structured_model.cache import build_cache
from task2_graph_editing.magicbrush.structured_model.evaluation import freeze_selection, summarize
from task2_graph_editing.magicbrush.structured_model.model import StructuredPredictor
from task2_graph_editing.magicbrush.structured_model.runtime import validate
from task2_graph_editing.magicbrush.structured_model.training import SELECTION_POLICY, QualityGateError, quality_gate, require_quality, train_one
from task2_graph_editing.magicbrush.structured_model.vocabulary import build_vocabulary, coverage, prototype_vectors

DEFAULT_CONFIG = ROOT / "configs/level2/magicbrush-manual-experiment-v3.json"


def training_queue(config):
    # Both architectures' development pilots precede any additional seed.
    first, *remaining = config["seeds"]
    return [(kind, first) for kind in config["models"]] + [
        (kind, seed) for kind in config["models"] for seed in remaining
    ]


def source_hashes():
    from common.provenance import source_hashes as current_sources
    return current_sources()


def versions():
    return {k: importlib.metadata.version(k) for k in ("torch", "numpy", "clip")}


def validate_config(config, root):
    if config["schema"] != "magicbrush-manual-experiment-v3" or config["device"] != "cuda":
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
        set(config["model"]) != {"hidden", "max_nodes", "max_edges", "layers", "add_slots", "dropout", "max_attributes"}
        or any(type(v) is not int or v <= 0 for k, v in config["model"].items() if k != "dropout")
        or not 0 <= config["model"]["dropout"] < 1
        or config["model"]["add_slots"] != 12
        or config["model"]["max_attributes"] != 32
        or config["model"]["max_nodes"] != 20
        or config["model"]["max_edges"] != 60
    ):
        raise ValueError("Invalid model architecture")
    for key, parent in (("text_cache", "data"), ("test_text_cache", "data"), ("output", "runs")):
        if not (root / config[key]).resolve().is_relative_to((root / parent).resolve()):
            raise ValueError("Output outside allowed root: " + key)
    for key in ("minimum_epochs", "validation_interval"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError("Invalid development budget: " + key)
    if config["minimum_epochs"] > config["epochs"] or config["validation_interval"] > config["patience"]:
        raise ValueError("Invalid warmup or validation interval")
    gate = config["quality_gate"]
    if set(gate) != {"minimum_valid_transaction_rate", "minimum_micro_f1"} or any(
        not isinstance(v, (int, float)) or isinstance(v, bool) or not 0 < v <= 1 for v in gate.values()
    ):
        raise ValueError("Expected positive predeclared validation quality thresholds")
    for key in ("text_cache", "test_text_cache", "output"):
        if "magicbrush-manual-experiment-v3" not in Path(config[key]).parts:
            raise ValueError("v3 must use isolated output/cache directories")
    if (
        config["loss_action_weight"] <= 0
        or len(config["action_class_weights"]) != 3
        or len(config["presence_class_weights"]) != 2
        or any(v <= 0 for v in config["action_class_weights"] + config["presence_class_weights"])
    ):
        raise ValueError("Invalid structured loss weights")
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
    vocab = build_vocabulary(examples["train"])
    report["vocabulary_coverage"] = {split: coverage(es, vocab) for split, es in examples.items()}
    if report["vocabulary_coverage"]["train"]["oov_payload_fields"]:
        raise ValueError("Training vocabulary does not cover its training labels")
    identity = {
        "vocabulary": vocab,
        "vocabulary_sha256": digest(vocab),
        "schema": config["schema"],
        "config": config,
        "release_lock_sha256": sha256(lock_path),
        "source_sha256": code,
        "versions": versions(),
        "train_ids": [e.id for e in examples["train"]],
        "val_ids": [e.id for e in examples["val"]],
        "human_review": report["human_review"],
        "model_inputs": "source graph and frozen CLIP inputs; output semantic prototypes derived from TRAIN labels only",
        "selection": SELECTION_POLICY,
        "test_targets_used_for_training_or_selection": False,
        "prior_test_exposure": "v1 evaluated this same test set; v3 development uses train/validation only",
        "parent_experiment": config["parent_experiment"],
    }
    report.update(
        stage="CPU preflight only",
        weights_sha256=config["weights_sha256"],
        experiment_sha256=digest(identity),
        code_and_environment_lock_verified=True,
        queue=[{"model": k, "seed": n} for k, n in training_queue(config)],
        quality_gate=config["quality_gate"],
        automatic_queue="both seed-42 pilots run; remaining seeds require both pilots to pass; all runs must pass before test",
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
        vocab = read(output / "experiment.json")["vocabulary"]
        model = StructuredPredictor(
            kind=selected["model"],
            vocabulary=vocab,
            prototypes=prototype_vectors(store, vocab),
            text_dim=store.dim,
            **config["model"],
        ).to(device)
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
            pilot_metrics = []
            for kind, seed in training_queue(config):
                directory = output / kind / str(seed)
                progress("pilot_train" if seed == config["seeds"][0] else "train", model=kind, seed=seed)
                if (directory / "completed-receipt.json").exists():
                    verify_receipt(directory, experiment_sha, kind, seed)
                else:
                    train_one(examples, store, config, training_identity, kind, seed, directory, wandb_mode=wandb_mode)
                    seal_run(directory, experiment_sha, kind, seed, examples["val"])
                result = read(directory / "result.json")
                gate = quality_gate(result["validation"], config)
                if gate != result["quality_gate"]:
                    raise ValueError("Stored quality gate differs from validation metrics")
                progress("validation_quality_gate", model=kind, seed=seed, **gate)
                if seed == config["seeds"][0]:
                    pilot_metrics.append(result["validation"])
                    if len(pilot_metrics) == len(config["models"]):
                        for metrics in pilot_metrics:
                            require_quality(metrics, config)
                else:
                    require_quality(result["validation"], config)
                gc.collect()
                torch.cuda.empty_cache()
            selection = freeze_selection(output, config, experiment_sha)
            selection_sha = sha256(output / "selection.json")
            progress("load_heldout_after_all_training")
            test, heldout_coverage = load_heldout(paths, config["expected_counts"]["test"], config["max_action_tokens"])
            test_identity = {
                "base_experiment_sha256": digest(identity),
                "selection_sha256": selection_sha,
                "scope": "test source inputs plus frozen TRAIN vocabulary prototypes after all checkpoint selection",
                "vocabulary": identity["vocabulary"],
                "vocabulary_sha256": identity["vocabulary_sha256"],
                "input_ids": [e.id for e in test],
            }
            write(output / "test-vocabulary-coverage.json", coverage(test, identity["vocabulary"]))
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
            summary = summarize(output, config, experiment_sha, examples["val"], test, heldout_coverage)
            lines = [
                "MagicBrush v3 completed; all six validation quality gates passed.",
                "Prior exposure: v1 already evaluated this test cohort; v3 development uses train/validation only.",
                "Metrics are against the frozen accepted AI-assisted labels; manual review covered 20 train cases.",
                f'Test coverage: {heldout_coverage["evaluated_pairs"]}/{heldout_coverage["original_test_pairs"]}; deferred: {heldout_coverage["deferred_pairs"]}.',
            ]
            for kind, splits in summary["models"].items():
                for split, result in splits.items():
                    metric = result["aggregate"]["micro"]["f1"]
                    lines.append(f'{kind} {split} micro F1: mean={metric["mean"]}, sample std={metric["std"]}')
            lines.append(
                "Full per-operation PRF1, exact match, failure rates, strata and no-edit baseline: summary.json"
            )
            fixed(output / "SUMMARY.txt", ("\n".join(lines) + "\n").encode())
            print("\n".join(lines), flush=True)
            progress("complete", validation_quality_gates_passed=True, summary=str(output / "summary.json"))
        except BaseException as error:
            progress(
                "interrupted"
                if isinstance(error, KeyboardInterrupt)
                else "quality_gate_failed"
                if isinstance(error, QualityGateError)
                else "failed",
                error=str(error),
                resume=(
                    "Quality failure is terminal for this frozen version; rerunning will not bypass the gate."
                    if isinstance(error, QualityGateError)
                    else "Rerun the same command; completed runs are verified and skipped, unfinished runs resume at saved epoch."
                ),
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
