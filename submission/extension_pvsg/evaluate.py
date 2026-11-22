"""Evaluate one PVSG checkpoint on internal validation or the held-out videos."""

import argparse
import json
from pathlib import Path

import torch
from extension_pvsg.m3.data import Store
from extension_pvsg.m3.models import RelationModel as M3
from extension_pvsg.m3.validation import validate as validate_m3
from extension_pvsg.m4.models import RelationModel as M4
from extension_pvsg.m4.validation import validate as validate_m4


class EvaluationStore(Store):
    def __init__(self, cache, features, split):
        super().__init__(cache, features)
        self.split = split

    def ids(self, split):
        return self.manifest["outer_split"]["test_ids"] if self.split == "test" else super().ids(split)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--family", choices=("m3", "m4"), required=True)
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
    model = (M3(identity["model"], cfg) if a.family == "m3" else M4(identity["model"], cfg)).to(a.device)
    model.load_state_dict(saved["model"])
    store = EvaluationStore(cfg["dataset_cache"], cfg["feature_cache"], a.split)
    validation = validate_m3 if a.family == "m3" else validate_m4
    print(json.dumps(validation(model, store, cfg, a.device, a.batch_size), indent=2))


if __name__ == "__main__":
    main()
