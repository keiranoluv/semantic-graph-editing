"""User-authorized sequential overnight matrix, with mandatory online W&B."""

import argparse
import gc
import json
import signal
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import torch

from experiments.aligng_gqa_v1 import runner as previous
from src.datasets.gqa_full import atomic_write, digest, pin_json
from src.knowledge.transe.core import sha256
from src.training.gqa_full import summarize_matrix
from tools.run_gqa_full import exclusive

from . import engine
from .batching import prepare_graph
from .model import FastConfig

ROOT = previous.ROOT
CONFIG = ROOT / "configs/level2/aligng-gqa-fast-v1-bs512.json"
PROBE = ROOT / "runs/level2/aligng-gqa-fast-v1/capacity.json"
MODELS = ("pe-plpr-aligng", "pe-plpr")
SEEDS = (42, 123, 3407)


def queue():
    return [(model, seed) for seed in SEEDS for model in MODELS]


def load_config(path=CONFIG):
    cfg = previous.load_config(path)
    if cfg["scope"] != "full" or cfg["train_batch_size"] != 512 or cfg["val_batch_size"] != 512:
        raise ValueError("This recipe requires full training with train/val batch512")
    if cfg["wandb"]["mode"] != "online":
        raise ValueError("Online W&B is mandatory")
    if cfg["output"] != "runs/level2/aligng-gqa-fast-v1/full":
        raise ValueError("Use the new isolated batch512 output, never a historical run directory")
    return cfg


def source_identity():
    base = previous.source_identity()
    files = [*Path(__file__).parent.glob("*.py"), ROOT / "scripts/run_aligng_gqa_fast.sh", CONFIG]
    names = sorted(str(p.relative_to(ROOT)) for p in files)
    for command in (
        ["git", "diff", "HEAD", "--", *names],
        ["git", "ls-files", "--others", "--exclude-standard", "--", *names],
    ):
        if subprocess.check_output(command, cwd=ROOT):
            raise ValueError("Commit the batch512 code before execution")
    return {
        **base,
        "parent_code_commit": base["code_commit"],
        "code_commit": subprocess.check_output(["git", "log", "-1", "--format=%H", "--", *names], cwd=ROOT)
        .decode()
        .strip(),
        "code_sha256": {**base["code_sha256"], **{str(p.relative_to(ROOT)): sha256(p) for p in files}},
    }


def prepare(cfg):
    exp, metadata, vectors = previous.prepare(cfg)
    exp.update(source_identity())
    exp["dimensions"]["fast"] = asdict(FastConfig())
    exp["execution_version"] = "aligng_fast_v1"
    exp["numerical_policy"] = "same equations; different dropout draw order and FP reduction; fresh experiment only"
    exp["config"] = {**cfg, "models": list(MODELS), "seeds": list(SEEDS)}
    exp.update(
        queue_order=[{"model": m, "seed": s} for m, s in queue()],
        queue_policy="one GPU model at a time; stop on error/signal; resume unfinished; verify and skip completed",
        config_sha256=sha256(CONFIG),
        capacity_cases_sha256=sha256(ROOT / "reports/level2/gqa-batch512-v1/benchmark.json"),
    )
    return exp, metadata, vectors


def verify_pilots():
    """Read-only verification of both historical pilot pairs; no rewriting reports."""
    for name in ("aligng-gqa-pilot-v1.json", "aligng-gqa-pilot-v1-bs16.json"):
        cfg = previous.load_config(ROOT / "configs/level2" / name)
        exp, _, _ = previous.prepare(cfg)
        directory = Path(cfg["output"])
        summary = json.loads((directory / "summary.json").read_text())
        if summary["experiment_sha256"] != digest(exp) or summary["status"] != "PASS":
            raise ValueError("Historical pilot identity mismatch")
        if not summary["matched_sampling"] or not summary["matched_initial_state"]:
            raise ValueError("Unmatched pilot controls")
        for model in MODELS:
            result = completed(exp, directory / model / "42", model, 42)
            if result is None or result != summary["results"][model]:
                raise ValueError("Missing/changed pilot result")


def verify_probe(exp, path=PROBE):
    result = json.loads(Path(path).read_text())
    if result["experiment_sha256"] != digest(exp) or result["status"] != "PASS":
        raise ValueError("Batch512 GPU probe missing/changed; do not start full training")
    required = {(m, c) for m in MODELS for c in ("train_max_candidates", "train_max_sampled", "val_max_candidates")}
    actual = {(r["model"], r["case"]) for r in result["results"] if r["status"] == "PASS" and r["images"] == 512}
    if actual != required:
        raise ValueError("Incomplete batch512 capacity evidence")
    for row in result["results"]:
        if row.get("variant") != "fast" or row.get("original", {}).get("status") != "PASS":
            raise ValueError("Missing original/fast comparison")
        if row["case"].startswith("val_") and row.get("eval_equivalence", {}).get("status") != "PASS":
            raise ValueError("Missing numerical equivalence evidence")


def completed(exp, directory, model, seed):
    path = Path(directory) / "result.json"
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    if result["identity"] != {"experiment_sha256": digest(exp), "model": model, "seed": seed}:
        raise ValueError("Completed run identity changed")
    if result["status"] != "PASS" or sha256(Path(directory) / "best.pt") != result["checkpoint_sha256"]:
        raise ValueError("Completed run checkpoint changed")
    return result


def run_queue(exp, train, val, vectors, *, stop_requested=lambda: False):
    """Caller holds GPU/output locks for the whole queue. No retry after failure."""
    output = Path(exp["config"]["output"])
    pin_json(output / "experiment.json", exp)
    state = {
        "experiment_sha256": digest(exp),
        "queue": exp["queue_order"],
        "completed": [],
        "status": "running",
        "current": None,
    }
    path = output / "queue-status.json"
    try:
        for model, seed in queue():
            if stop_requested():
                state["status"] = "interrupted"
                atomic_write(path, state)
                return 130
            state["current"] = {"model": model, "seed": seed}
            atomic_write(path, state)
            directory = output / model / str(seed)
            result = completed(exp, directory, model, seed)
            if result is None:
                print(f"[QUEUE START] {model} seed={seed} batch512/512", flush=True)
                result = engine.train_run(
                    model, seed, train, val, vectors, exp, directory, stop_requested=stop_requested
                )
                if result is None:
                    state["status"] = "interrupted"
                    atomic_write(path, state)
                    return 130
                if completed(exp, directory, model, seed) is None:
                    raise RuntimeError("Trainer returned without a durable completed result")
            else:
                print(f"[QUEUE SKIP] verified complete {model} seed={seed}", flush=True)
            state["completed"].append({"model": model, "seed": seed})
            atomic_write(path, state)
            gc.collect()
            if exp["config"]["device"] == "cuda":
                torch.cuda.empty_cache()
        summarize_matrix(exp, output)
        state.update(status="completed", current=None)
        atomic_write(path, state)
        return 0
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        atomic_write(path, state)
        raise


def datasets(cfg, metadata, workers=4):
    data = previous.datasets(cfg, metadata, workers)
    try:
        for split, dataset in zip(("train", "val"), data, strict=True):
            start = time.perf_counter()
            for i, graph in enumerate(dataset.graphs):
                dataset.graphs[i] = prepare_graph(graph)
                if (i + 1) % 10000 == 0:
                    print(f"[CPU PREP] {split}: {i + 1}/{len(dataset)} images", flush=True)
            print(f"[CPU PREP READY] {split}: {len(dataset)} images, {time.perf_counter() - start:.1f}s", flush=True)
        return data
    except BaseException:
        for dataset in data:
            dataset.close()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("plan", "check", "train-all", "summary"), default="plan")
    parser.add_argument("--loader-workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.loader_workers < 1:
        parser.error("Positive worker count required")
    cfg = load_config()
    exp, metadata, vectors = prepare(cfg)
    if args.stage == "plan":
        print(
            json.dumps(
                {"config": cfg, "experiment_sha256": digest(exp), "jobs": exp["queue_order"], "GPU_started": False},
                indent=2,
            )
        )
        return 0
    verify_pilots()
    verify_probe(exp)
    if args.stage == "check":
        with exclusive(ROOT / "runs/level2/.gqa-kg-training-device"):
            pass
        print(
            "[READY] six sequential jobs, train/val/effective batch512, online W&B, committed source verified",
            flush=True,
        )
        return 0
    if args.stage == "summary":
        summarize_matrix(exp, cfg["output"])
        return 0
    stopped = {"value": False}
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopped.update(value=True))
    with exclusive(ROOT / "runs/level2/.gqa-kg-training-device"), exclusive(cfg["output"]):
        engine.setup_runtime(cfg)
        pin_json(Path(cfg["output"]) / "experiment.json", exp)
        train, val = datasets(cfg, metadata, args.loader_workers)
        try:
            return run_queue(exp, train, val, vectors, stop_requested=lambda: stopped["value"])
        finally:
            train.close()
            val.close()


if __name__ == "__main__":
    raise SystemExit(main())
