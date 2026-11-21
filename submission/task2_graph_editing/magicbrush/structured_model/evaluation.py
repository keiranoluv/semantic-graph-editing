"""Freeze checkpoints only after every run clears its validation-only gate."""
from common.io import read, sha256, write

from task2_graph_editing.magicbrush.data_processing.evaluation import verify_receipt

from task2_graph_editing.magicbrush.structured_model.training import SELECTION_POLICY, require_quality


def freeze_selection(output, config, experiment_sha):
    checkpoints = []
    for kind in config["models"]:
        for seed in config["seeds"]:
            directory = output / kind / str(seed)
            receipt = verify_receipt(directory, experiment_sha, kind, seed)
            require_quality(read(directory / "result.json")["validation"], config)
            checkpoints.append(
                {
                    "model": kind,
                    "seed": seed,
                    "selected_epoch": receipt["selected_epoch"],
                    "checkpoint": str(directory / "best.pt"),
                    "checkpoint_sha256": receipt["files"]["best.pt"],
                    "completion_receipt_sha256": sha256(directory / "completed-receipt.json"),
                }
            )
    result = {
        "experiment_sha256": experiment_sha,
        "selection": SELECTION_POLICY,
        "test_metrics_available_at_selection": False,
        "prior_test_exposure": "v1 evaluated the same test cohort; v3 tuning uses only train/validation",
        "quality_gate": config["quality_gate"],
        "checkpoints": checkpoints,
    }
    write(output / "selection.json", result)
    return result


def summarize(output, config, experiment_sha, validation, test, coverage):
    from collections import Counter

    from task2_graph_editing.magicbrush.data_processing.evaluation import aggregate_metrics, score_rows
    from task2_graph_editing.magicbrush.data_processing.evaluation import summarize as base_summary

    summary = base_summary(output, config, experiment_sha, validation, test, coverage)
    diagnostics = {
        "protocol": "structured graph projection; raw decisions scored separately with same evaluator",
        "models": {},
    }
    for kind in config["models"]:
        diagnostics["models"][kind] = {}
        for split, examples in (("val", validation), ("test", test)):
            metrics, constraints = [], Counter()
            for seed in config["seeds"]:
                rows = read(output / kind / str(seed) / f"predictions-{split}.json")
                raw = [{"id": r["id"], "tokens": r["raw_tokens"], "error": r["raw_error"]} for r in rows]
                metrics.append(score_rows(examples, raw))
                for row in rows:
                    constraints.update(row["constraints"])
            diagnostics["models"][kind][split] = {
                "raw_aggregate": aggregate_metrics(metrics),
                "constraint_counts_all_seeds": dict(constraints),
            }
    write(output / "raw-and-constraint-diagnostics.json", diagnostics)
    return summary
