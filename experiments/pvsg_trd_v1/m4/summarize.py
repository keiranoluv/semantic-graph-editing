"""Internal-validation summaries only; incomplete seeds stay explicitly missing."""

import argparse
from pathlib import Path

import numpy as np

from ..contracts import read_json
from ..m2.download import sha256
from ..m3.common import write_json
from ..validate_m0 import canonical_sha256
from .common import config, identity


def summarize(cfg, kind):
    rows, missing, reference_feature_sha = [], [], None
    for seed in cfg["seeds"]:
        out = Path(cfg["output_root"]) / kind / f"seed-{seed}"
        if not (out / "complete.json").is_file():
            missing.append(seed)
            continue
        completed, run = read_json(out / "complete.json"), read_json(out / "run.json")
        ident = run["identity"]
        feature_sha = ident["feature_summary_sha256"]
        if reference_feature_sha is not None and feature_sha != reference_feature_sha:
            raise ValueError("Cannot aggregate different feature caches")
        reference_feature_sha = feature_sha
        expected = identity(cfg, feature_sha)
        if (
            any(ident[key] != value for key, value in expected.items())
            or ident["model"] != kind
            or ident["seed"] != seed
            or canonical_sha256(ident) != run["signature"]
            or completed["signature"] != run["signature"]
            or completed["test_evaluated"]
            or sha256(out / "best.pt") != completed["best_checkpoint_sha256"]
            or sha256(out / "history.json") != completed["history_sha256"]
        ):
            raise ValueError(f"Invalid completed run identity/checkpoint: {out}")
        report = read_json(out / "best_validation.json")
        key = report["metrics"]["0.5"]["50"]
        if [key["mR"], key["R"]] != completed["best_key"]:
            raise ValueError("Best-validation metrics differ from completion record")
        rows.append(dict(seed=seed, mR50=key["mR"], R50=key["R"], wandb_url=completed["wandb_url"]))
    return dict(
        model=kind,
        scope="internal validation; not held-out test or upstream PVSG benchmark",
        runs=rows,
        missing_seeds=missing,
        complete_three_seeds=not missing,
        aggregate={
            name: {
                "mean": float(np.mean([r[name] for r in rows])),
                "sample_std": float(np.std([r[name] for r in rows], ddof=1)) if len(rows) > 1 else None,
            }
            for name in ("mR50", "R50")
        }
        if rows
        else None,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(config()["models"]), required=True)
    args = parser.parse_args()
    cfg = config()
    report = summarize(cfg, args.model)
    path = Path(cfg["output_root"]) / args.model / "summary.json"
    write_json(path, report)
    print(f"{path}: {len(report['runs'])}/3 seeds; missing={report['missing_seeds']}")


if __name__ == "__main__":
    main()
