"""Six new GQA runs only; reuse audited historical controls without retraining."""

import argparse
import gc
import json
import signal
import subprocess
from pathlib import Path

import torch

from experiments.aligng_fast_v1 import runner as parent
from experiments.aligng_gqa_v1 import runner as base
from src.datasets.gqa_full import atomic_write, digest, pin_json
from src.knowledge.transe.core import sha256
from src.training.gqa_full import summarize_matrix
from tools.run_gqa_full import exclusive

from . import engine

ROOT = parent.ROOT
CONFIG = ROOT / "configs/level2/aligng-ablation-v1-bs512.json"
PROBE = ROOT / "runs/level2/aligng-ablation-v1/capacity.json"
MODELS = engine.MODELS
SEEDS = (42, 123, 3407)
CASES = ROOT / "reports/level2/gqa-batch512-v1/benchmark.json"


def queue():
    return [(m, s) for s in SEEDS for m in MODELS]


def load_config(path=CONFIG):
    cfg = base.load_config(path)
    expected = parent.load_config()
    expected["output"] = "runs/level2/aligng-ablation-v1/full"
    if cfg != expected:
        raise ValueError("Ablation must preserve historical batch512 recipe except isolated output")
    return cfg


def source_identity():
    identity = parent.source_identity()
    files = [*Path(__file__).parent.glob("*.py"), CONFIG, ROOT / "scripts/run_aligng_ablation_v1.sh"]
    names = sorted(str(p.relative_to(ROOT)) for p in files)
    for command in (
        ["git", "diff", "HEAD", "--", *names],
        ["git", "ls-files", "--others", "--exclude-standard", "--", *names],
    ):
        if subprocess.check_output(command, cwd=ROOT):
            raise ValueError("Commit ablation code/config before execution")
    return {
        **identity,
        "parent_code_commit": identity["code_commit"],
        "code_commit": subprocess.check_output(["git", "log", "-1", "--format=%H", "--", *names], cwd=ROOT)
        .decode()
        .strip(),
        "code_sha256": {**identity["code_sha256"], **{str(p.relative_to(ROOT)): sha256(p) for p in files}},
    }


def controls(cfg):
    """Verify completed baselines and matching data/optimization; never train controls."""
    recipes = {
        "pe-ce": ROOT / "runs/level2/gqa-full-v3-bs512",
        "pe-plpr": ROOT / "runs/level2/aligng-gqa-fast-v1/full",
        "pe-plpr-aligng": ROOT / "runs/level2/aligng-gqa-fast-v1/full",
    }
    rows = []
    for model, folder in recipes.items():
        historical = json.loads((folder / "experiment.json").read_text())
        keys = (
            "manifest",
            "cache",
            "glove",
            "epochs",
            "train_batch_size",
            "val_batch_size",
            "lr",
            "weight_decay",
            "gradient_clip_norm",
            "early_stopping",
        )
        if any(historical["config"][k] != cfg[k] for k in keys):
            raise ValueError(f"Historical control recipe mismatch: {model}")
        for seed in SEEDS:
            directory = folder / model / str(seed)
            result = parent.completed(historical, directory, model, seed)
            if result is None:
                raise ValueError("Historical control missing")
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "result": str((directory / "result.json").relative_to(ROOT)),
                    "result_sha256": sha256(directory / "result.json"),
                    "checkpoint_sha256": result["checkpoint_sha256"],
                    "experiment_sha256": digest(historical),
                }
            )
    return rows


def prepare(cfg):
    exp, metadata, vectors = parent.prepare(cfg)
    exp.update(source_identity())
    exp.update(
        execution_version="aligng_ablation_v1",
        config={**cfg, "models": list(MODELS), "seeds": list(SEEDS)},
        variant_flags=engine.VARIANTS,
        queue_order=[{"model": m, "seed": s} for m, s in queue()],
        config_sha256=sha256(CONFIG),
        historical_controls=controls(cfg),
        control_note="PE-CE uses historical PE-Net engine; common recipe/features/loss; not identical dropout trajectory to active feedback.",
    )
    return exp, metadata, vectors


def verify_probe(exp, path=PROBE):
    data = json.loads(Path(path).read_text())
    required = {(m, c) for m in MODELS for c in ("train_max_candidates", "train_max_sampled", "val_max_candidates")}
    actual = {(r["model"], r["case"]) for r in data["results"] if r["status"] == "PASS" and r["images"] == 512}
    if (
        data["status"] != "PASS"
        or data["experiment_sha256"] != digest(exp)
        or actual != required
        or data["case_source_sha256"] != sha256(CASES)
    ):
        raise ValueError("Missing/changed GPU batch512 capacity evidence")


def run_queue(exp, train, val, vectors, *, stop_requested=lambda: False):
    output = Path(exp["config"]["output"])
    pin_json(output / "experiment.json", exp)
    state = {"experiment_sha256": digest(exp), "queue": exp["queue_order"], "completed": [], "status": "running"}
    try:
        for model, seed in queue():
            if stop_requested():
                state["status"] = "interrupted"
                atomic_write(output / "queue-status.json", state)
                return 130
            state["current"] = {"model": model, "seed": seed}
            atomic_write(output / "queue-status.json", state)
            directory = output / model / str(seed)
            result = parent.completed(exp, directory, model, seed)
            if result is None:
                print(f"[QUEUE START] {model} seed={seed} train/val batch512", flush=True)
                result = engine.train_run(
                    model, seed, train, val, vectors, exp, directory, stop_requested=stop_requested
                )
                if result is None:
                    state["status"] = "interrupted"
                    atomic_write(output / "queue-status.json", state)
                    return 130
                if parent.completed(exp, directory, model, seed) is None:
                    raise RuntimeError("Missing durable result")
            else:
                print(f"[QUEUE SKIP] verified {model} seed={seed}", flush=True)
            state["completed"].append({"model": model, "seed": seed})
            atomic_write(output / "queue-status.json", state)
            gc.collect()
            if exp["config"]["device"] == "cuda":
                torch.cuda.empty_cache()
        summarize_matrix(exp, output)
        state.update(status="completed", current=None)
        atomic_write(output / "queue-status.json", state)
        return 0
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        atomic_write(output / "queue-status.json", state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("plan", "check", "train-all", "summary"), default="plan")
    parser.add_argument("--loader-workers", type=int, default=4)
    args = parser.parse_args()
    if args.loader_workers < 1:
        parser.error("Positive worker count required")
    cfg = load_config()
    exp, metadata, vectors = prepare(cfg)
    if args.stage == "plan":
        print(
            json.dumps(
                {"experiment_sha256": digest(exp), "jobs": exp["queue_order"], "controls": exp["historical_controls"]},
                indent=2,
            )
        )
        return 0
    verify_probe(exp)
    if args.stage == "check":
        with exclusive(ROOT / "runs/level2/.gqa-kg-training-device"):
            pass
        print("[READY] six NEW jobs; batch512; online W&B; historical controls verified")
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
        train, val = parent.datasets(cfg, metadata, args.loader_workers)
        try:
            return run_queue(exp, train, val, vectors, stop_requested=lambda: stopped["value"])
        finally:
            train.close()
            val.close()


if __name__ == "__main__":
    raise SystemExit(main())
