"""Audit selected checkpoints on CPU, then evaluate an immutable test queue."""
import argparse
import gc
import hashlib
import json
import statistics
from pathlib import Path

import torch

from common.datasets.gqa_full import GQAFullDataset, atomic_write, digest, pin_json, verify_cache
from common.knowledge.transe.core import sha256
from common.training.gqa_full import create_model, load_vectors
from task2_gqa.gqa_test_v1.cache import TestDataset, prepare_cache
from task2_gqa.run_gqa_full import exclusive

MANIFEST = Path("data/processed/level2/gqa-v1/manifest.json")
OUTPUT = Path("reports/level2/gqa-test-v1")
SOURCES = [
    ("gqa-full-v3-bs512", "base", ["mlp", "gat", "pe-ce", "pe-pl", "pe-plpr"]),
    ("aligng-gqa-fast-v1/full", "fast", ["pe-plpr", "pe-plpr-aligng"]),
    ("aligng-ablation-v1/full", "ablation", ["edge-only", "pe-ce-aligng"]),
    ("gqa-wordnet-full-v2-bs512", "kg", ["gat-kg"]),
]
SEEDS = [42, 123, 3407]
CACHE_CONFIG = {
    "scope": "full",
    "root": "data/raw/gqa",
    "manifest": str(MANIFEST),
    "weights": "data/cache/vg150-clip-vit-b32-v1/weights/ViT-B-32.pt",
    "cache": "data/cache/gqa-test-clip-v1",
    "reference_cache": "data/cache/gqa-full-clip-v1",
    "cache_shard_size": 256,
    "clip_batch_size": 64,
    "device": "cuda",
}


def read(path):
    return json.loads(Path(path).read_text())


def state_hash(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        # Feedback is disabled in the duplicated fast PE-PLPR control.
        if key.startswith("feedback."):
            continue
        tensor = value.detach().cpu().contiguous()
        h.update(key.encode())
        h.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def compact(metrics):
    return {
        mode: {
            key: metrics[mode][("ng-" if mode == "ng" else "") + key] for key in ("R@50", "R@100", "mR@50", "mR@100")
        }
        for mode in ("gc", "ng")
    }


def inventory():
    jobs, aliases, audit, code = [], [], [], {}
    for folder, backend, models in SOURCES:
        root = Path("runs/level2") / folder
        experiment = read(root / "experiment.json")
        assets = dict(experiment.get("assets", {}))
        if "manifest_sha256" in experiment:
            assets[str(MANIFEST)] = experiment["manifest_sha256"]
            assets["data/processed/level2/gqa-glove-v1/manifest.json"] = experiment["glove_manifest_sha256"]
        if backend == "kg":
            assets["data/knowledge/gqa-wordnet-features-v2/manifest.json"] = experiment["feature_manifest_sha256"]
            assets["data/knowledge/gqa-wordnet-features-v2/class_embeddings.pt"] = experiment["features_tensor_sha256"]
        for name, expected in assets.items():
            if sha256(Path(name)) != expected:
                raise ValueError(f"Historical asset changed: {name}")
        for name, expected in experiment["code_sha256"].items():
            if sha256(Path(name)) != expected:
                raise ValueError(f"Historical source changed: {name}")
            code[name] = expected
        for kind in models:
            for seed in SEEDS:
                directory = root / kind / str(seed)
                result = read(directory / "result.json")
                identity = {"experiment_sha256": digest(experiment), "model": kind, "seed": seed}
                checkpoint = directory / "best.pt"
                checksum = sha256(checkpoint)
                if result["status"] != "PASS" or result["identity"] != identity:
                    raise ValueError(f"Unfinished/mismatched run: {directory}")
                if checksum != result["checkpoint_sha256"]:
                    raise ValueError(f"Checkpoint changed: {directory}")
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                history = read(directory / "history.json")
                selected = max(history, key=lambda r: r["validation"]["mR@50"])
                if (
                    payload["identity"] != identity
                    or payload["epoch"] != result["selected_epoch"]
                    or selected["epoch"] != result["selected_epoch"]
                    or len(history) != result["epochs_completed"]
                ):
                    raise ValueError(f"Selection/history mismatch: {directory}")
                digest(history)  # rejects NaN/Inf anywhere in training history
                if payload["score"] != selected["validation"]["mR@50"]:
                    raise ValueError("Checkpoint selection score mismatch")
                if result["validation"]["gc"]["mR@50"] != payload["score"]:
                    raise ValueError("Reported validation differs from selected checkpoint")
                if backend == "ablation":
                    control = read(Path("runs/level2/gqa-full-v3-bs512/pe-plpr") / str(seed) / "history.json")
                    if any(
                        a["sampling_trace_sha256"] != b["sampling_trace_sha256"]
                        for a, b in zip(history, control, strict=False)
                    ):
                        raise ValueError("Ablation sampling differs from historical control")
                    patience = experiment["config"]["early_stopping"]["patience"]
                    if result["stopping_reason"] == "early_stopping" and len(history) - payload["epoch"] != patience:
                        raise ValueError("Early stopping patience differs")
                    from task2_gqa.ablation.engine import VARIANTS

                    if result["variant_flags"] != VARIANTS[kind]:
                        raise ValueError("Ablation flags differ")
                    if kind == "pe-ce-aligng" and any(
                        epoch["train_losses"][loss] != 0
                        for epoch in history
                        for loss in ("pl_euclidean", "pr_cosine", "pr_euclidean")
                    ):
                        raise ValueError("Unexpected auxiliary loss in CE-only run")
                row = {
                    "model": kind,
                    "seed": seed,
                    "backend": backend,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": checksum,
                    "result_sha256": sha256(directory / "result.json"),
                    "history_sha256": sha256(directory / "history.json"),
                    "experiment_sha256": digest(experiment),
                    "selected_epoch": payload["epoch"],
                    "epochs_completed": len(history),
                    "stopping_reason": result["stopping_reason"],
                    "validation": compact(result["validation"]),
                    "identity": identity,
                }
                audit.append(row)
                if kind == "pe-plpr":
                    row["effective_state_sha256"] = state_hash(payload["model"])
                if backend == "fast" and kind == "pe-plpr":
                    original = next(j for j in jobs if j["model"] == kind and j["seed"] == seed)
                    if row["effective_state_sha256"] != original["effective_state_sha256"]:
                        raise ValueError("PE-PLPR control differs; do not deduplicate")
                    aliases.append(
                        {
                            "source": row,
                            "canonical_checkpoint": original["checkpoint"],
                            "reason": "Identical active PE-PLPR tensors; feedback disabled",
                        }
                    )
                else:
                    jobs.append(row)
    for name in sorted(Path("task2_gqa/gqa_test_v1").glob("*.py")):
        code[str(name)] = sha256(name)
    return {
        "schema_version": 1,
        "split": "project held-out test from GQA official val",
        "manifest_sha256": sha256(MANIFEST),
        "test_images": len(read(MANIFEST)["splits"]["test"]),
        "jobs": jobs,
        "aliases": aliases,
        "code_sha256": code,
        "reserved": ["aligng-residual", "aligng-ema"],
        "audit": audit,
    }


def build(job):
    if sha256(Path(job["checkpoint"])) != job["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed after preflight")
    payload = torch.load(job["checkpoint"], map_location="cpu", weights_only=False)
    dims = payload["dimensions"]
    config = {"manifest": str(MANIFEST), "glove": "data/processed/level2/gqa-glove-v1"}
    vectors = load_vectors(config, dims)
    kind, backend = job["model"], job["backend"]
    if backend == "kg":
        from task2_gqa.gqa_wordnet_v2_features import load_v2
        from task2_gqa.run_gqa_kg_pilot import create_model as factory

        features, valid, _ = load_v2(Path("data/knowledge/gqa-wordnet-features-v2"), MANIFEST)
        model = factory(kind, dims, features, valid)
    elif backend in ("fast", "ablation"):
        if backend == "fast":
            from task2_gqa.training.engine import create_model as factory
        else:
            from task2_gqa.ablation.engine import create_model as factory
        model = factory(kind, dims, vectors)
    else:
        model = create_model(kind, dims, vectors)
    model.load_state_dict(payload["model"], strict=True)
    return model.eval(), dims


def evaluate(job, graphs, device, batch_size):
    model, dims = build(job)
    if job["backend"] in ("fast", "ablation"):
        from task2_gqa.training.evaluation import evaluate as evaluator
    else:
        from task2_gqa.train_gqa_pilot import evaluate as evaluator
    try:
        return evaluator(
            model.to(device),
            "gat" if job["backend"] == "kg" else job["model"],
            graphs,
            {**dims, "batch_size": batch_size},
            device,
            include_ng=True,
        )
    finally:
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()


def summarize(rows):
    summary = {}
    for kind in sorted({r["model"] for r in rows}):
        group = [r for r in rows if r["model"] == kind]
        summary[kind] = {"seeds": [r["seed"] for r in group]}
        for mode in ("gc", "ng"):
            summary[kind][mode] = {}
            for metric in ("R@50", "R@100", "mR@50", "mR@100"):
                values = [r["metrics"][mode][("ng-" if mode == "ng" else "") + metric] for r in group]
                summary[kind][mode][metric] = {
                    "values": values,
                    "mean": statistics.mean(values),
                    "std_sample": statistics.stdev(values) if len(values) > 1 else None,
                }
    return summary


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=["audit", "cpu-smoke", "run"])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    plan = inventory()
    if args.stage == "audit":
        atomic_write(args.output / "training-audit.json", plan)
        print(f"PASS: {len(plan['audit'])} sources, {len(plan['jobs'])} unique jobs, {len(plan['aliases'])} aliases")
        return
    if args.stage == "cpu-smoke":
        metadata = read("data/cache/gqa-full-clip-v1/manifest.json")
        dataset = GQAFullDataset("data/cache/gqa-full-clip-v1", metadata, "val")
        try:
            graphs = dataset[:2]
            rows = []
            for job in plan["jobs"] + [a["source"] for a in plan["aliases"]]:
                metrics = evaluate(job, graphs, "cpu", 2)
                digest(metrics)
                rows.append(
                    {"model": job["model"], "seed": job["seed"], "backend": job["backend"], "metrics": compact(metrics)}
                )
                print(f"CPU PASS {job['backend']}/{job['model']}/{job['seed']}", flush=True)
            for alias in plan["aliases"]:
                seed = alias["source"]["seed"]
                matched = [r["metrics"] for r in rows if r["model"] == "pe-plpr" and r["seed"] == seed]
                if matched[0] != matched[1]:
                    raise ValueError("Duplicated control smoke predictions differ")
            atomic_write(
                args.output / "cpu-smoke.json",
                {"status": "PASS", "split": "validation", "images": dataset.ids[:2], "jobs": rows, "test_used": False},
            )
        finally:
            dataset.close()
        return
    with exclusive(args.output), exclusive(Path(CACHE_CONFIG["cache"])):
        pin_json(args.output / "plan.json", plan)
        prepare_cache(CACHE_CONFIG)
        metadata = verify_cache(CACHE_CONFIG["cache"])
        if metadata["identity"]["splits"] != {"test": read(MANIFEST)["splits"]["test"]}:
            raise ValueError("Test cache split mismatch")
        identity = {
            "plan_sha256": digest(plan),
            "cache_manifest_sha256": sha256(Path(CACHE_CONFIG["cache"]) / "manifest.json"),
        }
        dataset, rows = TestDataset(CACHE_CONFIG["cache"], metadata), []
        try:
            for index, job in enumerate(plan["jobs"], 1):
                path = args.output / job["model"] / str(job["seed"]) / "result.json"
                expected = {**identity, "checkpoint_sha256": job["checkpoint_sha256"]}
                if path.exists():
                    row = read(path)
                    if (
                        row["identity"] != expected
                        or row["status"] != "PASS"
                        or row["metrics_sha256"] != digest(row["metrics"])
                    ):
                        raise ValueError(f"Invalid existing result: {path}")
                    print(f"[SKIP {index}/27] {path}", flush=True)
                else:
                    print(f"[RUN {index}/27] {job['model']}/{job['seed']}", flush=True)
                    metrics = evaluate(job, dataset, "cuda", args.batch_size)
                    row = {
                        "status": "PASS",
                        "model": job["model"],
                        "seed": job["seed"],
                        "identity": expected,
                        "test_images": len(dataset),
                        "metrics": metrics,
                        "batch_size": args.batch_size,
                        "device": "cuda",
                        "torch_version": str(torch.__version__),
                        "metrics_sha256": digest(metrics),
                    }
                    atomic_write(path, row)
                rows.append(row)
                atomic_write(
                    args.output / "summary.json",
                    {
                        "complete": len(rows) == len(plan["jobs"]),
                        "completed": len(rows),
                        "expected": len(plan["jobs"]),
                        "models": summarize(rows),
                        "identity": identity,
                    },
                )
        finally:
            dataset.close()


if __name__ == "__main__":
    main()
