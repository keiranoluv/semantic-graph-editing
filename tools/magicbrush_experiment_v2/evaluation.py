"""Freeze checkpoints only after every run clears its validation-only gate."""
from tools.emnlp2020_graph_modification_v1.dataset import read, sha256, write
from tools.magicbrush_experiment_v1.evaluation import verify_receipt

from .training import SELECTION_POLICY, require_quality


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
        "prior_test_exposure": "v1 evaluated the same test cohort; v2 tuning uses only train/validation",
        "quality_gate": config["quality_gate"],
        "checkpoints": checkpoints,
    }
    write(output / "selection.json", result)
    return result
