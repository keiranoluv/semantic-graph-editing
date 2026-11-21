"""Prediction receipts and multi-seed metrics, retaining unsupported-class counts."""
from statistics import mean, stdev

from src.editing.operations import KINDS
from tools.edit_model_v1.codec import encode_actions, score_predictions
from tools.emnlp2020_graph_modification_v1.dataset import read, sha256, write


def score_rows(examples, rows):
    if [r["id"] for r in rows] != [e.id for e in examples]:
        raise ValueError("Prediction IDs/order differ from evaluation cohort")
    metrics, errors = score_predictions(examples, [r["tokens"] for r in rows])
    if errors != [r["error"] for r in rows]:
        raise ValueError("Stored prediction errors differ from decoded outputs")
    metrics["failed_prediction_rate"] = metrics["failed_predictions"] / len(examples) if examples else None
    metrics["zero_support_types"] = [k for k in KINDS if metrics["per_type"][k]["support"] == 0]
    return metrics


def strata(examples, predictions):
    result = {}
    for group in ("add", "delete", "replace", "attribute_change"):
        indices = []
        for i, e in enumerate(examples):
            if group == "attribute_change":
                selected = any(
                    o.kind == "replace_node"
                    and o.before.label == o.after.label
                    and o.before.attributes != o.after.attributes
                    for o in e.operations
                )
            else:
                selected = any(o.kind.startswith(group + "_") for o in e.operations)
            if selected:
                indices.append(i)
        result[group] = score_rows([examples[i] for i in indices], [predictions[i] for i in indices])
    return result


def no_edit(examples):
    rows = [{"id": e.id, "tokens": encode_actions(())[1:], "error": None} for e in examples]
    return score_rows(examples, rows)


def seal_run(directory, experiment_sha, kind, seed, validation):
    result = read(directory / "result.json")
    expected = {"experiment_sha256": experiment_sha, "model": kind, "seed": seed}
    if result["identity"] != expected:
        raise ValueError("Training result identity mismatch")
    for field, name in (("best_checkpoint_sha256", "best.pt"), ("last_checkpoint_sha256", "last.pt")):
        if result[field] != sha256(directory / name):
            raise ValueError("Training checkpoint hash mismatch")
    predictions = read(directory / "predictions-val.json")
    scored = score_rows(validation, predictions)
    if any(scored[k] != v for k, v in result["validation"].items()):
        raise ValueError("Validation predictions/metrics mismatch")
    receipt = {
        "identity": expected,
        "selected_epoch": result["selected_epoch"],
        "files": {
            name: sha256(directory / name) for name in ("best.pt", "last.pt", "result.json", "predictions-val.json")
        },
    }
    write(directory / "completed-receipt.json", receipt)
    return receipt


def verify_receipt(directory, experiment_sha, kind, seed):
    receipt = read(directory / "completed-receipt.json")
    if receipt["identity"] != {"experiment_sha256": experiment_sha, "model": kind, "seed": seed}:
        raise ValueError("Completed receipt belongs to another experiment")
    required = {"best.pt", "last.pt", "result.json", "predictions-val.json"}
    if set(receipt["files"]) != required:
        raise ValueError("Incomplete receipt")
    for name, checksum in receipt["files"].items():
        if sha256(directory / name) != checksum:
            raise ValueError("Completed run was modified")
    return receipt


def freeze_selection(output, config, experiment_sha):
    # No test-label read is allowed before this function succeeds for all six runs.
    selected = []
    for kind in config["models"]:
        for seed in config["seeds"]:
            directory = output / kind / str(seed)
            receipt = verify_receipt(directory, experiment_sha, kind, seed)
            selected.append(
                {
                    "model": kind,
                    "seed": seed,
                    "selected_epoch": receipt["selected_epoch"],
                    "checkpoint": str(directory / "best.pt"),
                    "checkpoint_sha256": receipt["files"]["best.pt"],
                    "completion_receipt_sha256": sha256(directory / "completed-receipt.json"),
                }
            )
    selection = {
        "experiment_sha256": experiment_sha,
        "selection": "validation micro operation F1 only; ties earlier epoch",
        "test_metrics_available_at_selection": False,
        "checkpoints": selected,
    }
    write(output / "selection.json", selection)
    return selection


def stats(values):
    present = [v for v in values if v is not None]
    return {
        "mean": mean(present) if present else None,
        "std": stdev(present) if len(present) > 1 else None,
        "n_defined": len(present),
        "n_total": len(values),
        "std_definition": "sample standard deviation, ddof=1",
    }


def aggregate_metrics(metrics):
    result = {
        k: stats([r[k] for r in metrics])
        for k in (
            "exact_operation_set_accuracy",
            "simple_exact_operation_set_accuracy",
            "exact_target_graph_accuracy",
            "valid_transaction_rate",
            "failed_prediction_rate",
            "macro_f1_supported_types",
        )
    }
    result["micro"] = {k: stats([r["micro"][k] for r in metrics]) for k in ("precision", "recall", "f1")}
    result["per_type"] = {}
    for kind in KINDS:
        supports = [r["per_type"][kind]["support"] for r in metrics]
        if len(set(supports)) != 1:
            raise ValueError("Seeds were evaluated on different label cohorts")
        result["per_type"][kind] = {
            "support_per_seed": supports[0],
            **{
                k: stats([r["per_type"][kind][k] if supports[0] else None for r in metrics])
                for k in ("precision", "recall", "f1")
            },
            "false_positive_counts": [r["per_type"][kind]["fp"] for r in metrics],
        }
    return result


def summarize(output, config, experiment_sha, validation, test, coverage):
    summary = {
        "experiment_sha256": experiment_sha,
        "seeds": config["seeds"],
        "test_coverage": coverage,
        "simple_definition": "exactly one GT operation, graph-level proxy for a simple request",
        "strata_definition": "overlapping subsets based on GT operation types; attribute change is same-label replace_node",
        "zero_support_policy": "per-type PRF1 shown as undefined when no GT support; false positives retained separately",
        "label_scope": "AI-assisted observed image changes; 20 train examples manually checked, not all test labels human GT",
        "no_edit": {"val": no_edit(validation), "test": no_edit(test)},
        "models": {},
    }
    for kind in config["models"]:
        by_split = {"val": [], "test": []}
        by_group = {split: {g: [] for g in ("add", "delete", "replace", "attribute_change")} for split in by_split}
        for seed in config["seeds"]:
            directory = output / kind / str(seed)
            verify_receipt(directory, experiment_sha, kind, seed)
            evaluation = read(directory / "test-result.json")
            if evaluation["experiment_sha256"] != experiment_sha or evaluation["selection_sha256"] != sha256(
                output / "selection.json"
            ):
                raise ValueError("Test result identity mismatch")
            if evaluation["predictions_sha256"] != sha256(directory / "predictions-test.json"):
                raise ValueError("Test predictions changed")
            for split, examples in (("val", validation), ("test", test)):
                predictions = read(directory / f"predictions-{split}.json")
                scored = score_rows(examples, predictions)
                if split == "test" and scored != evaluation["metrics"]:
                    raise ValueError("Test metrics differ from predictions")
                by_split[split].append(scored)
                for group, metrics in strata(examples, predictions).items():
                    by_group[split][group].append(metrics)
        summary["models"][kind] = {
            split: {
                "aggregate": aggregate_metrics(by_split[split]),
                "per_seed": dict(zip(map(str, config["seeds"]), by_split[split], strict=True)),
                "by_edit_type": {g: aggregate_metrics(v) for g, v in by_group[split].items()},
            }
            for split in by_split
        }
    write(output / "summary.json", summary)
    return summary
