"""Train one PVSG model with the batch and schedule used in the report."""

import argparse
import os
from pathlib import Path

import torch
from common.locking import exclusive
from common.paths import ROOT
from common.provenance import source_hashes
from extension_pvsg.baselines.preflight import check_features, check_runtime
from extension_pvsg.baselines.runner import train
from extension_pvsg.cases import CASES
from extension_pvsg.contracts import read_json


def main():
    base = read_json(ROOT / "extension_pvsg/baselines/configs/baselines_v1.json")
    ablations = read_json(ROOT / "extension_pvsg/ablations/configs/ablations_v1.json")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=[*base["models"], *ablations["models"]], required=True)
    parser.add_argument("--seed", type=int, choices=(17, 42, 2026), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-epochs", type=int, choices=(10, 50), default=10)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.minimum_epochs == 50 and args.seed not in CASES.get(args.model, ()):
        parser.error("The reported min50 follow-up covers only the seven model/seed pairs in cases.py")
    family = "baselines" if args.model in base["models"] else "ablations"
    cfg = dict(base if family == "baselines" else ablations)
    cfg.update(
        max_epochs=150,
        min_epochs=args.minimum_epochs,
        early_stopping_patience=15,
        experiment_id=f"pvsg-submission-{family}-min{args.minimum_epochs}",
        output_root=str(args.output_dir),
    )
    os.chdir(ROOT)
    source_hashes()
    check_runtime()
    feature_hash = check_features(cfg)
    if args.check:
        print(f"Ready {args.model}, seed={args.seed}, batch=32, max_epochs=150, min_epochs={args.minimum_epochs}")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for PVSG training")
    os.environ["WANDB_MODE"] = "online"
    with exclusive(ROOT / "runs/pvsg-training-gpu"):
        train(cfg, args.model, args.seed, 32, feature_hash, torch.device("cuda:0"))


if __name__ == "__main__":
    main()
