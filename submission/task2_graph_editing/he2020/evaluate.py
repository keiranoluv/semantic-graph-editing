"""Authorized evaluation of six frozen v4 checkpoints. Never trains or initializes W&B."""

import argparse
import os
import subprocess
from pathlib import Path

import torch

from task2_graph_editing.magicbrush.sequence_model.codec import EOS, score_predictions
from common.io import digest, read, sha256, write
from task2_graph_editing.he2020.preprocessing.dataset import ROOT
from task2_graph_editing.he2020.models.runner import evaluate, inputs_tensor, make_model, predict, retrieval_baseline
from task2_graph_editing.he2020.data_processing.data import load_bundle
from task2_graph_editing.he2020.data_processing.model import ActionVocabulary, frozen_features
from task2_graph_editing.he2020.report import aggregate, sanitized_cases
from task2_graph_editing.he2020.summarize import audit, verify_hashes

DEFAULT_OUTPUT = Path("runs/level2/emnlp2020-task-b-wandb-v4-test-v1")
DEFAULT_REPORT = Path("reports/level2/emnlp2020-task-b-wandb-v4-test-v1")


def prepare_inputs(config):
    runs, _, hashes, experiment = audit(config)
    dataset = Path(config["dataset"])
    receipt_path = dataset / "test-receipt.json"
    verify_hashes(dataset, read(receipt_path), hashes)
    examples = load_bundle(dataset, "test")["test"]
    if not examples or len({e.id for e in examples}) != len(examples):
        raise ValueError("empty_or_duplicate_test_examples")
    reserved = read(dataset / "trainval-groups.json")
    if any(e.group in reserved for e in examples):
        raise ValueError("test_source_overlap_with_frozen_trainval_groups")
    directory = dataset / "clip-test"
    if not (directory / "manifest.json").is_file() or not (directory / "vectors.pt").is_file():
        raise ValueError("requires_existing_frozen_test_features_no_implicit_feature_inference")
    store = frozen_features(
        [e.model_input() for e in examples], directory, config["weights"], config["text_batch_size"]
    )
    if read(directory / "manifest.json")["target_text_encoded"]:
        raise ValueError("test_target_features_forbidden")
    vocabulary = ActionVocabulary(experiment["vocabulary"])
    tensors = inputs_tensor(examples, store, config)
    evidence = {
        "training_freeze_sha256": sha256(Path(config["output"]) / "frozen-checkpoints.json"),
        "experiment_identity": experiment["identity"],
        "test_receipt_sha256": sha256(receipt_path),
        "test_features_manifest_sha256": sha256(directory / "manifest.json"),
        "test_features_vectors_sha256": sha256(directory / "vectors.pt"),
    }
    return runs, examples, vocabulary, store, tensors, evidence


def evaluation_identity(config, device, evidence):
    return {
        "schema": "emnlp2020-wandb-v4-test-v1",
        "training_config": config,
        "device": device,
        "torch": str(torch.__version__),
        "TF32": False,
        "deterministic_algorithms": True,
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        "eval_code_sha256": sha256(Path(__file__)),
        "audit_code_sha256": sha256(ROOT / "task2_graph_editing/he2020/summarize.py"),
        **evidence,
        "test_previously_evaluated_in_v3": True,
        "training_performed": False,
        "WandB_initialized": False,
    }


def checked_evaluation(path, identity_sha256, checkpoint_sha256):
    verify_hashes(path, read(path / "completed-receipt.json"), {})
    result = read(path / "result.json")
    if result["evaluation_identity_sha256"] != identity_sha256 or result["checkpoint_sha256"] != checkpoint_sha256:
        raise ValueError("completed_test_identity_or_checkpoint_changed")
    return result


def evaluate_test(config, output, *, device="cuda"):
    output = Path(output)
    if output.resolve() in {
        Path(config["output"]).resolve(),
        Path(config["dataset"]).resolve(),
        (ROOT / "runs/level2/emnlp2020-task-b-gpu-v3").resolve(),
    }:
        raise ValueError("evaluation_requires_new_output_preserve_training_artifacts")
    # Audit every frozen run, completion receipt and source hash before test/model execution.
    runs, examples, vocabulary, store, tensors, evidence = prepare_inputs(config)
    identity = evaluation_identity(config, device, evidence)
    identity_sha256 = digest(identity)
    path = output / "identity.json"
    if path.exists():
        if read(path)["identity"] != identity:
            raise ValueError("evaluation_code_inputs_or_device_changed_use_new_output")
    else:
        write(
            path,
            {
                "identity": identity,
                "identity_sha256": identity_sha256,
                "creation_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            },
        )
    runtime_config = {**config, "device": device}
    results = []
    for frozen in runs:
        name = f'{frozen["model"]}-seed{frozen["seed"]}'
        source = Path(config["output"]) / name
        checkpoint_sha256 = sha256(source / "best.pt")
        if checkpoint_sha256 != read(source / "result.json")["checkpoint_sha256"]:
            raise ValueError("checkpoint_changed_after_audit")
        destination = output / name
        if (destination / "completed-receipt.json").exists():
            results.append(checked_evaluation(destination, identity_sha256, checkpoint_sha256))
            continue
        torch.manual_seed(frozen["seed"])
        checkpoint = torch.load(source / "best.pt", map_location="cpu", weights_only=True)
        model = make_model(runtime_config, frozen["model"], vocabulary)
        model.load_state_dict(checkpoint["state_dict"])
        predictions = predict(model, tensors, vocabulary, runtime_config)
        metrics, cases, selected = evaluate(examples, predictions, vocabulary)
        del model, checkpoint
        if device == "cuda":
            torch.cuda.empty_cache()
        write(destination / "predictions.json", cases)
        write(destination / "cases.json", selected)
        result = {
            "model": frozen["model"],
            "seed": frozen["seed"],
            "checkpoint_sha256": checkpoint_sha256,
            "evaluation_identity_sha256": identity_sha256,
            "metrics": metrics,
            "predictions_sha256": sha256(destination / "predictions.json"),
            "test_was_previously_evaluated_in_v3": True,
        }
        write(destination / "result.json", result)
        write(
            destination / "completed-receipt.json",
            {"files": {n: sha256(destination / n) for n in ("result.json", "predictions.json", "cases.json")}},
        )
        results.append(result)
        print(
            f"Evaluated {name}: test={len(examples)}, F1={metrics['micro']['f1']:.6f}, exact={metrics['exact_operation_set_accuracy']:.6f}",
            flush=True,
        )
    baseline_path = output / "baselines.json"
    if not baseline_path.exists():
        no_op = score_predictions(examples, [[ord("[") + 3, ord("]") + 3, EOS]] * len(examples))[0]
        nearest = retrieval_baseline(config, examples, store)
        write(baseline_path, {"no_op": no_op, "CLIP_nearest_train_program": nearest})
    summary = {
        "runs": results,
        "examples": len(examples),
        "split": "official_test_source_disjoint_subset",
        "baselines": read(baseline_path),
        "evaluation_identity_sha256": identity_sha256,
        "model_inference_performed": True,
        "training_performed": False,
        "WandB_initialized": False,
        "automatic_training_queue": False,
        "test_previously_seen_in_v3": True,
    }
    write(output / "summary.json", summary)
    write(
        output / "completed-receipt.json",
        {"files": {n: sha256(output / n) for n in ("identity.json", "summary.json", "baselines.json")}},
    )
    return summary


def summarize(output, report):
    output, report = Path(output), Path(report)
    verify_hashes(output, read(output / "completed-receipt.json"), {})
    summary = read(output / "summary.json")
    info = read(output / "identity.json")
    for run in summary["runs"]:
        checked_evaluation(
            output / f'{run["model"]}-seed{run["seed"]}',
            summary["evaluation_identity_sha256"],
            run["checkpoint_sha256"],
        )
    metrics = {
        kind: aggregate([r["metrics"] for r in summary["runs"] if r["model"] == kind]) for kind in ("mlp", "gnn")
    }
    comparisons = {}
    errors = {}
    for run in summary["runs"]:
        name = f'{run["model"]}-seed{run["seed"]}'
        old = Path("runs/level2/emnlp2020-task-b-gpu-v3") / name
        previous = read(old / "test-result.json")
        if previous["test_receipt_sha256"] != info["identity"]["test_receipt_sha256"]:
            raise ValueError("v3_test_comparison_requires_same_test_receipt")
        comparisons[name] = {
            "metrics_identical_to_v3": run["metrics"] == previous["metrics"],
            "prediction_bytes_identical_to_v3": run["predictions_sha256"] == sha256(old / "test-predictions.json"),
        }
        errors[name] = sanitized_cases(read(output / name / "cases.json"))
    write(
        report / "metrics.json",
        {
            "models": metrics,
            "runs": summary["runs"],
            "baselines": summary["baselines"],
            "examples": summary["examples"],
            "test_previously_seen_in_v3": True,
            "comparison_with_v3": comparisons,
            "fresh_GPU_evaluation_of_v4_checkpoints": info["identity"]["device"] == "cuda",
            "evaluation_device": info["identity"]["device"],
        },
    )
    write(report / "provenance.json", info)
    write(report / "error-analysis.json", errors)
    write(
        report / "artifact-receipt.json",
        {
            "files": {
                str(output / name): checksum
                for name, checksum in read(output / "completed-receipt.json")["files"].items()
            },
            "per_run_receipts": {
                f'{r["model"]}-seed{r["seed"]}': sha256(
                    output / f'{r["model"]}-seed{r["seed"]}' / "completed-receipt.json"
                )
                for r in summary["runs"]
            },
        },
    )
    print(f"Wrote test statistics to {report}; no GPU workload in summarize stage.")


def main():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/level2/emnlp2020-task-b-wandb-v4.json"))
    parser.add_argument("--stage", choices=("eval", "summarize"), required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.stage == "summarize":
        summarize(args.output, args.report)
        return
    config = read(args.config)
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA_unavailable_no_silent_CPU_fallback")
    evaluate_test(config, args.output, device=args.device)


if __name__ == "__main__":
    main()
