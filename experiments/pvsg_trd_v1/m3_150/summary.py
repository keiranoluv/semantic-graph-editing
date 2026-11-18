"""Summarize all nine selected internal-val results, never as held-out metrics."""

from pathlib import Path

import numpy as np

from ..contracts import read_json
from ..m2.download import sha256
from ..m3.common import write_json


def summarize(cfg):
    root = Path(cfg["output_root"])
    result = dict(scope="best_internal_validation_only_not_test", models={}, test_evaluated=False)
    for model in cfg["models"]:
        rows = []
        for seed in cfg["seeds"]:
            path = root / model / f"seed-{seed}"
            complete = read_json(path / "complete.json")
            if sha256(path / "best.pt") != complete["best_checkpoint_sha256"]:
                raise ValueError("Best checkpoint hash changed")
            report = read_json(path / "best_validation.json")
            primary = report["metrics"]["0.5"]["50"]
            rows.append(
                dict(
                    seed=seed,
                    R50=primary["R"],
                    mR50=primary["mR"],
                    epochs=complete["epochs"],
                    wandb_url=complete["wandb_url"],
                    checkpoint_sha256=complete["best_checkpoint_sha256"],
                )
            )
        result["models"][model] = dict(
            runs=rows,
            aggregation={
                key: dict(
                    mean=float(np.mean([r[key] for r in rows])),
                    std_sample=float(np.std([r[key] for r in rows], ddof=1)),
                )
                for key in ("R50", "mR50")
            },
        )
    write_json(root / "validation_summary.json", result)
    print("All nine runs summarized; held-out test has not been evaluated")
