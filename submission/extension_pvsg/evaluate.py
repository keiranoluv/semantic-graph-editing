"""Evaluate one PVSG checkpoint on internal validation or the held-out videos."""

import argparse
import json
from pathlib import Path

import torch
from extension_pvsg.ablations.models import RelationModel as AblationModel
from extension_pvsg.ablations.models import collate as collate_ablations
from extension_pvsg.baselines.data import Store
from extension_pvsg.baselines.models import RelationModel as BaselineModel
from extension_pvsg.baselines.validation import validate


class EvaluationStore(Store):
    def __init__(self, cache, features, split):
        super().__init__(cache, features)
        self.split = split

    def ids(self, split):
        return self.manifest["outer_split"]["heldout_ids"] if self.split == "test" else super().ids(split)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", choices=("validation", "test"), default="validation")
    p.add_argument("--allow-test", action="store_true")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--batch-size", type=int, default=1)
    a = p.parse_args()
    if a.split == "test" and not a.allow_test:
        p.error("Held-out test requires --allow-test")
    if a.batch_size < 1:
        p.error("Batch size must be positive")
    saved = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    identity = saved["identity"]
    cfg = identity["config"]
    ablation = "variants" in cfg
    model = (AblationModel if ablation else BaselineModel)(identity["model"], cfg).to(a.device)
    model.load_state_dict(saved["model"])
    store = EvaluationStore(cfg["dataset_cache"], cfg["feature_cache"], a.split)
    options = {"collate_fn": collate_ablations} if ablation else {}
    print(json.dumps(validate(model, store, cfg, a.device, a.batch_size, **options), indent=2))


if __name__ == "__main__":
    main()
