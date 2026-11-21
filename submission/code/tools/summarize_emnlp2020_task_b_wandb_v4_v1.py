"""CPU-only audit/summary of user-completed v4 artifacts. No model inference or W&B API."""

import argparse
import csv
import io
import math
from collections import Counter
from pathlib import Path

import torch

from src.editing.operations import Operation
from tools.edit_model_v1.codec import encode_actions
from tools.emnlp2020_graph_modification_v1.dataset import digest, encode, fixed, read, sha256, write
from tools.emnlp2020_task_b_gpu_v3.runner import evaluate
from tools.emnlp2020_task_b_v2.data import load_bundle
from tools.emnlp2020_task_b_v2.model import ActionVocabulary
from tools.emnlp2020_task_b_wandb_v4.runner import identity
from tools.emnlp2020_task_b_wandb_v4.tracking import flatten
from tools.emnlp2020_task_b_wandb_v4.training import advance, initial_state, should_stop
from tools.report_emnlp2020_task_b_v2 import aggregate, classify, sanitized_cases


def check_wandb_summary(result, summary):
    """selected_val belongs to best.pt; val/* belongs to the final epoch."""
    expected = {
        "best_epoch": result["best_epoch"],
        "epochs_completed": result["epochs_completed"],
        "best_val_loss": result["best_val_loss"],
        **flatten(result["validation"], "selected_val"),
    }
    for key, value in expected.items():
        if key not in summary or not math.isclose(summary[key], value, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("WandB_selected_summary_mismatch:" + key)
    if summary.get("stopping_reason") != result["stopping_reason"]:
        raise ValueError("WandB_stopping_reason_mismatch")


def check_stopping(result, config):
    state = initial_state()
    history = result["history"]
    if len(history) != result["epochs_completed"]:
        raise ValueError("incomplete_epoch_history")
    for index, row in enumerate(history, 1):
        if row["epoch"] != index or should_stop(state, config):
            raise ValueError("history_continued_after_early_stop_or_out_of_order")
        state, _ = advance(state, row["val/loss"], config)
        if row["early_stop/bad_epochs"] != state["bad_epochs"] or row["early_stop/best_epoch"] != state["best_epoch"]:
            raise ValueError("early_stop_history_mismatch")
    if result["best_epoch"] != state["best_epoch"] or result["best_val_loss"] != state["best_loss"]:
        raise ValueError("wrong_selected_epoch")
    reason = "early_stopping" if should_stop(state, config) else "epoch_budget"
    if result["stopping_reason"] != reason or state["epoch"] > config["epochs"]:
        raise ValueError("stopping_reason_mismatch")
    if reason == "epoch_budget" and state["epoch"] != config["epochs"]:
        raise ValueError("incomplete_epoch_budget")
    return state


def verify_hashes(directory, receipt, seen):
    for name, checksum in receipt["files"].items():
        path = directory / name
        if sha256(path) != checksum:
            raise ValueError("artifact_changed:" + str(path))
        seen[str(path)] = checksum


def audit(config):
    root, dataset = Path(config["output"]), Path(config["dataset"])
    experiment, freeze = read(root / "experiment.json"), read(root / "frozen-checkpoints.json")
    if identity(config) != experiment["source"]:
        raise ValueError("trained_source_or_config_changed")
    for name, path in {
        "trainval": dataset / "trainval-receipt.json",
        "features": dataset / "clip-trainval/manifest.json",
    }.items():
        if sha256(path) != experiment["inputs"][name]:
            raise ValueError("experiment_input_changed")
    if digest([experiment["source"], experiment["inputs"], experiment["vocabulary"]]) != experiment["identity"]:
        raise ValueError("experiment_identity_changed")
    expected = {(k, s) for k in config["models"] for s in config["seeds"]}
    if len(freeze["runs"]) != len(expected) or {(r["model"], r["seed"]) for r in freeze["runs"]} != expected:
        raise ValueError("incomplete_or_duplicate_queue")
    if (
        freeze["experiment_identity"] != experiment["identity"]
        or not freeze["all_runs_tracked_online"]
        or freeze["test_used"]
    ):
        raise ValueError("invalid_training_freeze")
    seen = {str(root / n): sha256(root / n) for n in ("experiment.json", "frozen-checkpoints.json")}
    receipt = read(dataset / "trainval-receipt.json")
    verify_hashes(dataset, receipt, seen)
    features = read(dataset / "clip-trainval/manifest.json")
    if sha256(dataset / "clip-trainval/vectors.pt") != features["vectors_sha256"]:
        raise ValueError("feature_vectors_changed")
    examples = load_bundle(dataset, "trainval")["val"]
    by_id = {e.id: e for e in examples}
    vocabulary = ActionVocabulary(experiment["vocabulary"])
    runs, details = [], {}
    for frozen in freeze["runs"]:
        name = f'{frozen["model"]}-seed{frozen["seed"]}'
        path = root / name
        for file, key in (
            ("best.pt", "checkpoint_sha256"),
            ("result.json", "result_sha256"),
            ("completed-receipt.json", "completed_receipt_sha256"),
        ):
            if sha256(path / file) != frozen[key]:
                raise ValueError("frozen_run_changed:" + name)
            seen[str(path / file)] = frozen[key]
        verify_hashes(path, read(path / "completed-receipt.json"), seen)
        result = read(path / "result.json")
        run_identity = {"experiment_identity": experiment["identity"], "model": frozen["model"], "seed": frozen["seed"]}
        if (
            result["run_identity"] != run_identity
            or read(path / "identity.json") != run_identity
            or result["test_used"]
        ):
            raise ValueError("run_identity_or_scope_changed")
        state = check_stopping(result, config)
        best = torch.load(path / "best.pt", weights_only=True, map_location="cpu")
        last = torch.load(path / "last.pt", weights_only=True, map_location="cpu")
        if (
            best["epoch"] != result["best_epoch"]
            or best["config"] != config
            or best["vocabulary"] != vocabulary.tokens
            or best["experiment_identity"] != experiment["identity"]
        ):
            raise ValueError("best_checkpoint_metadata_changed")
        if last["stop_state"] != state or last["history"] != result["history"] or last["run_identity"] != run_identity:
            raise ValueError("last_checkpoint_state_changed")
        if not all(torch.equal(value, last["best_state_dict"][key]) for key, value in best["state_dict"].items()):
            raise ValueError("last_selected_weights_differ_from_best")
        wb = read(path / "wandb-run.json")
        if (
            wb != result["wandb"]
            or wb != frozen["wandb"]
            or wb["mode"] != "online"
            or wb["id"] != digest(run_identity)[:20]
        ):
            raise ValueError("WandB_identity_mismatch")
        candidates = list((root / "wandb").glob(f'run-*-{wb["id"]}/files/wandb-summary.json'))
        if len(candidates) != 1:
            raise ValueError("missing_or_multiple_local_WandB_summaries")
        check_wandb_summary(result, read(candidates[0]))
        seen[str(candidates[0])] = sha256(candidates[0])
        cases = read(path / "val-predictions.json")
        if (
            len(cases) != len(examples)
            or len({c["id"] for c in cases}) != len(cases)
            or {c["id"] for c in cases} != set(by_id)
        ):
            raise ValueError("prediction_coverage_changed")
        ordered, tokens = [], []
        for case in cases:
            e = by_id[case["id"]]
            if (
                encode(case["source"]) != encode(e.source.to_dict())
                or encode(case["target"]) != encode(e.target.to_dict())
                or encode(case["expected"]) != encode([o.to_dict() for o in e.operations])
                or case["instruction"] != e.instruction
            ):
                raise ValueError("saved_prediction_reference_changed")
            ordered.append(e)
            # Preserve original parse failures; [] actions would be a valid no-op.
            tokens.append(
                []
                if case["error"] and case["error"].startswith("parse:")
                else encode_actions(tuple(Operation.from_dict(o) for o in case["predicted"]))[1:]
            )
        rescored, _, _ = evaluate(ordered, tokens, vocabulary)
        if rescored != result["validation"]:
            raise ValueError("saved_prediction_metrics_mismatch:" + name)
        previous = Path("runs/level2/emnlp2020-task-b-gpu-v3") / name
        old_result = read(previous / "result.json")
        if sha256(previous / "best.pt") != old_result["checkpoint_sha256"]:
            raise ValueError("previous_checkpoint_changed")
        old_best = torch.load(previous / "best.pt", weights_only=True, map_location="cpu")
        same_weights = old_best["state_dict"].keys() == best["state_dict"].keys() and all(
            torch.equal(value, old_best["state_dict"][key]) for key, value in best["state_dict"].items()
        )
        runs.append(
            {
                "model": result["model"],
                "seed": result["seed"],
                "epochs_completed": result["epochs_completed"],
                "best_epoch": result["best_epoch"],
                "best_val_loss": result["best_val_loss"],
                "stopping_reason": result["stopping_reason"],
                "runtime_seconds": result["runtime_seconds"],
                "validation": result["validation"],
                "wandb": wb,
                "same_selected_parameters_as_v3": same_weights,
                "same_validation_prediction_bytes_as_v3": sha256(previous / "val-predictions.json")
                == sha256(path / "val-predictions.json"),
                "validation_metrics_equal_v3": result["validation"] == old_result["validation"],
                "WandB_selected_summary_verified": True,
                "saved_predictions_rescored_on_CPU": True,
            }
        )
        details[name] = {
            "categories": dict(Counter(classify(c) for c in cases)),
            "selected_cases": sanitized_cases(read(path / "val-cases.json")),
            "history": result["history"],
        }
    return runs, details, seen, experiment


def report(config_path, output):
    config, output = read(config_path), Path(output)
    runs, details, seen, experiment = audit(config)
    models = {kind: aggregate([r["validation"] for r in runs if r["model"] == kind]) for kind in config["models"]}
    total_epochs = sum(r["epochs_completed"] for r in runs)
    write(
        output / "summary.json",
        {
            "split": "validation",
            "examples": runs[0]["validation"]["examples"],
            "models": models,
            "runs": runs,
            "total_epochs": total_epochs,
            "previous_v3_total_epochs": 72,
            "epoch_reduction_vs_v3_fraction": 1 - total_epochs / 72,
            "all_selected_parameters_identical_to_v3": all(r["same_selected_parameters_as_v3"] for r in runs),
            "test_results_included": False,
            "GPU_workloads_performed_by_report": 0,
            "WandB_evidence_scope": "local online run records, local selected_val summary and launcher sync logs; no remote API query",
        },
    )
    write(
        output / "error-analysis.json",
        {name: {k: v for k, v in detail.items() if k != "history"} for name, detail in details.items()},
    )
    write(
        output / "provenance.json",
        {
            "config": config,
            "creation_commit": experiment["creation_commit"],
            "experiment_identity": experiment["identity"],
            "trained_source": experiment["source"]["files"],
            "artifact_hashes": seen,
            "report_script_sha256": sha256(Path(__file__)),
            "no_artifacts_modified": True,
        },
    )
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(
        [
            "model",
            "seed",
            "epoch",
            "train_loss",
            "val_loss",
            "val_micro_f1",
            "val_exact_match",
            "bad_epochs",
            "selected_epoch",
            "final_epoch",
        ]
    )
    for run in runs:
        for row in details[f'{run["model"]}-seed{run["seed"]}']["history"]:
            writer.writerow(
                [
                    run["model"],
                    run["seed"],
                    row["epoch"],
                    row["train/loss"],
                    row["val/loss"],
                    row["val/micro/f1"],
                    row["val/exact_operation_set_accuracy"],
                    row["early_stop/bad_epochs"],
                    row["epoch"] == run["best_epoch"],
                    row["epoch"] == run["epochs_completed"],
                ]
            )
    fixed(output / "learning-curves.csv", stream.getvalue().encode())
    print(f"Verified and rescored {len(runs)} runs on CPU; {total_epochs} epochs; validation only; report: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/level2/emnlp2020-task-b-wandb-v4.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/level2/emnlp2020-task-b-wandb-v4-results-v1"))
    args = parser.parse_args()
    report(args.config, args.output)
