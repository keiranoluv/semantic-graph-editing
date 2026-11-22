"""Small shared utilities, experiment identity and atomic artifacts."""

import json
import random
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from extension_pvsg.m2.download import sha256

ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    # CUDA scatter reductions can be nondeterministic; record this limitation.
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False


def code_identity():
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(ROOT.rglob("*.py")) if not p.name.startswith("test_")}


def runtime_versions():
    return {name: version(name) for name in ("torch", "torch-geometric", "numpy", "wandb", "protobuf", "setuptools")}


def identity(cfg, feature_summary_sha256):
    return {
        "config": cfg,
        "code": code_identity(),
        "runtime": runtime_versions(),
        "m2_status_sha256": sha256(ROOT.parent / "m2/provenance/m2_status.json"),
        "feature_summary_sha256": feature_summary_sha256,
        "split_sha256": sha256(ROOT.parent / "m2/manifests/dataset.lock.json"),
    }
