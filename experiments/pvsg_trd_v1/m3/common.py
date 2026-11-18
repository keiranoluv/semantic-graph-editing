"""Small shared utilities, experiment identity and atomic artifacts."""

import json
import random
import subprocess
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from ..contracts import read_json
from ..m2.download import sha256
from ..validate_m0 import canonical_sha256

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
CONFIG_PATH = ROOT / "configs/baselines_v1.json"


def config():
    cfg = read_json(CONFIG_PATH)
    if cfg["pos_weight"] != 1.0 or cfg["decoder_threshold"] != 0.5 or cfg["knowledge_graph"]:
        raise ValueError("This M3 version implements unweighted BCE, frozen .5 decoder and no KG")
    if cfg["hidden_dim"] % cfg["heads"] or cfg["precision"] != "bf16":
        raise ValueError("Unsupported model dimensions or precision")
    if any(cfg["effective_batch_windows"] % size for size in cfg["profile_batch_candidates"]):
        raise ValueError("Every physical batch candidate must divide the effective batch")
    return cfg


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


def require_committed():
    state = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", str(ROOT)], cwd=PROJECT, text=True
    )
    if state.strip():
        raise ValueError("Commit M3 code before a GPU run")
    subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(ROOT / "models.py")],
        cwd=PROJECT,
        check=True,
        stdout=subprocess.DEVNULL,
    )


def signature(cfg, feature_sha):
    return canonical_sha256(identity(cfg, feature_sha))
