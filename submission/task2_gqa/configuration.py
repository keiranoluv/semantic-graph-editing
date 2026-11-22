"""One explicit custom-GQA run per invocation; full training requires online W&B."""

import copy
import importlib.metadata
import json
from dataclasses import asdict
from pathlib import Path

from common.datasets.gqa_full import GQAFullDataset, digest, verify_cache
from common.datasets.gqa_pilot import validate_cohort
from common.knowledge.transe.core import sha256
from common.paths import ROOT
from common.training.gqa_full import load_vectors, math_finite, model_config
from task2_gqa.dataset import RAMGQADataset
from task2_gqa.models.model import AlignGConfig


def load_config(path):
    cfg = json.loads(Path(path).read_text())
    expected = {
        "schema_version",
        "scope",
        "manifest",
        "cache",
        "glove",
        "cohort",
        "output",
        "epochs",
        "train_batch_size",
        "val_batch_size",
        "lr",
        "weight_decay",
        "gradient_clip_norm",
        "checkpoint_every_batches",
        "device",
        "num_threads",
        "early_stopping",
        "wandb",
        "aligng",
    }
    if set(cfg) != expected or cfg["schema_version"] != 1 or cfg["scope"] not in ("pilot", "full"):
        raise ValueError("Invalid versioned AlignG configuration")
    for key in ("epochs", "train_batch_size", "val_batch_size", "checkpoint_every_batches", "num_threads"):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(f"Invalid {key}")
    if cfg["device"] not in ("cpu", "cuda") or cfg["wandb"]["mode"] not in ("disabled", "online"):
        raise ValueError("Invalid device/tracker mode")
    if cfg["scope"] == "full" and cfg["wandb"]["mode"] != "online":
        raise ValueError("Full training requires W&B online; no disabled/offline fallback")
    if cfg["scope"] == "pilot" and cfg["epochs"] > 5:
        raise ValueError("P1 pilot is bounded to at most five epochs")
    if any(not math_finite(cfg[k]) for k in ("lr", "weight_decay", "gradient_clip_norm")):
        raise ValueError("Non-finite optimization settings")
    if cfg["lr"] <= 0 or cfg["weight_decay"] < 0 or cfg["gradient_clip_norm"] <= 0:
        raise ValueError("Invalid optimization settings")
    early = cfg["early_stopping"]
    if set(early) != {"enabled", "patience", "min_delta"} or type(early["enabled"]) is not bool:
        raise ValueError("Invalid early stopping")
    if type(early["patience"]) is not int or early["patience"] < 1 or early["min_delta"] != 0:
        raise ValueError("Expected positive patience and min_delta=0")
    if cfg["aligng"] != asdict(AlignGConfig(**cfg["aligng"])):
        raise ValueError("Pin all adapter flags explicitly")
    if cfg["aligng"]["loss_mode"] != "plpr" or not cfg["aligng"]["feedback_enabled"]:
        raise ValueError("Recipe is PE-PLPR; --model explicitly selects feedback on/off")
    return cfg


def source_identity():
    from common.provenance import revision, source_hashes

    provenance = json.loads((ROOT / "task2_gqa/models/provenance.json").read_text())
    code_revision = revision()
    return {
        "code_commit": code_revision,
        "code_sha256": source_hashes(),
        "upstream_commit": provenance["upstream_commit"],
    }


def prepare(cfg):
    """CPU-only asset verification; never instantiate CLIP or open test annotations."""
    provenance = json.loads((ROOT / "task2_gqa/models/provenance.json").read_text())
    pinned = provenance["audited_metadata_sha256"]
    for path in (cfg["manifest"], str(Path(cfg["cache"]) / "manifest.json"), str(Path(cfg["glove"]) / "manifest.json")):
        if path not in pinned or sha256(Path(path)) != pinned[path]:
            raise ValueError(f"Unpinned/changed input metadata: {path}")
    manifest = json.loads(Path(cfg["manifest"]).read_text())
    meta = verify_cache(cfg["cache"])
    expected = {k: manifest["splits"][k] for k in ("train", "val")}
    if meta["identity"]["splits"] != expected or meta["identity"]["manifest_sha256"] != sha256(Path(cfg["manifest"])):
        raise ValueError("Cache split/manifest mismatch")
    if cfg["scope"] == "pilot":
        if cfg["cohort"] not in pinned or sha256(Path(cfg["cohort"])) != pinned[cfg["cohort"]]:
            raise ValueError("Changed pilot cohort")
        cohort = json.loads(Path(cfg["cohort"]).read_text())
        validate_cohort(cohort, manifest, sha256(Path(cfg["manifest"])))
        selected = cohort["splits"]
    else:
        selected = expected
    dimensions = {**model_config(manifest), "aligng": cfg["aligng"]}
    if dimensions["num_classes"] != 311 or dimensions["num_object_classes"] != 1700:
        raise ValueError("Expected custom GQA 310 foreground/1699 objects")
    vectors = load_vectors(cfg, dimensions)
    exp = {
        "schema_version": 1,
        "dataset": "custom GQA; not GQA200",
        "config": cfg,
        "dimensions": dimensions,
        **source_identity(),
        "assets": {p: pinned[p] for p in pinned if p.startswith("data/")},
        "glove_vectors_sha256": sha256(Path(cfg["glove"]) / "vectors.npz"),
        "cohort": selected,
        "cohort_sha256": digest(selected),
        "protocol": "unchanged GQA GC/NG reference IoU50; all 310 predicates; duplicate GT rows",
        "selection": "max validation GC mR@50; earliest tie; patience15 for full",
        "sampling": "all positives + 3 negatives/positive; no-GT up to32; frozen sampler RNG",
        "effective_batch_size": cfg["train_batch_size"],
        "gradient_accumulation": 1,
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "torch-geometric", "numpy", "wandb")},
        "adaptation": "Frozen-CLIP PredCls pair proxy; per-image full label-free context; global classifier; P0 README deviations",
        "test_used": False,
    }
    subset = copy.deepcopy(meta)
    subset["identity"]["splits"] = selected
    return exp, subset, vectors


def datasets(cfg, metadata, workers=4):
    if cfg["scope"] == "full":
        return [RAMGQADataset(cfg["cache"], metadata, k, workers=workers) for k in ("train", "val")]
    groups = []
    for split in ("train", "val"):
        dataset = GQAFullDataset(cfg["cache"], metadata, split)
        try:
            groups.append([dataset[i] for i in range(len(dataset))])
        finally:
            dataset.close()
    return groups
