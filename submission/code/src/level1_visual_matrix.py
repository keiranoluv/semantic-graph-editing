"""Run the matched CLIP-feature PredCls matrix from the user's GPU shell."""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from src.level1_matrix import describe, run_parallel

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache/vg150-clip-vit-b32-v1"
KG = ROOT / "data/knowledge/c2-wordnet-reviewed-filtered-v2"
RUNS = ROOT / "runs/level1-clip-v1"
REPORTS = ROOT / "reports/level1"
LOGS = ROOT / "logs/level1-clip-v1"
SEEDS = (42, 123, 3407)
CONDITIONS = ("mlp_semantic", "gat", "kg_on", "kg_off", "edges_off")
SUMMARY = REPORTS / "clip-v1-matched-3seed-val5000-summary.json"


def name(condition: str, seed: int) -> str:
    return f"clip-{condition}-seed{seed}-v1"


def run_dir(condition: str, seed: int) -> Path:
    folder = RUNS / name(condition, seed)
    return folder if condition in {"kg_on", "kg_off", "edges_off"} else folder / "run_0001"


def report_path(condition: str, seed: int) -> Path:
    return REPORTS / f"clip-v1-{condition}-seed{seed}-val5000.json"


def common_args(seed: int, wandb_mode: str) -> list[str]:
    return [
        "--root",
        str(ROOT / "data/raw/vg150"),
        "--visual-cache",
        str(CACHE),
        "--device",
        "cuda",
        "--seed",
        str(seed),
        "--epochs",
        "200",
        "--patience",
        "15",
        "--min-delta",
        "0",
        "--batch-size",
        "512",
        "--val-batch-size",
        "512",
        "--num-workers",
        "0",
        "--max-train-images",
        "57723",
        "--max-val-images",
        "5000",
        "--hidden-dim",
        "256",
        "--embedding-dim",
        "32",
        "--lr",
        "0.001",
        "--weight-decay",
        "0.01",
        "--negative-ratio",
        "3",
        "--box-policy",
        "clip",
        "--wandb-mode",
        wandb_mode,
    ]


def train_command(condition: str, seed: int, wandb_mode: str) -> list[str]:
    base = [sys.executable, "-u", "-m"]
    common = common_args(seed, wandb_mode)
    if condition == "mlp_semantic":
        return (
            base
            + ["scripts.pilot_mlp", "--model", "mlp_semantic"]
            + common
            + ["--output-dir", str(RUNS), "--run-name", name(condition, seed)]
        )
    gat = ["--accumulation-steps", "1", "--num-heads", "4", "--num-layers", "2", "--dropout", "0.1"]
    if condition == "gat":
        return (
            base
            + ["scripts.train_gat"]
            + common
            + gat
            + ["--output-dir", str(RUNS), "--run-name", name(condition, seed)]
        )
    command = (
        base
        + ["scripts.train_gat_kg", "--kg-features", str(KG)]
        + common
        + gat
        + ["--output-dir", str(run_dir(condition, seed))]
    )
    if condition == "kg_off":
        command.append("--no-knowledge")
    elif condition == "edges_off":
        command.extend(("--message-edges", "none"))
    elif condition != "kg_on":
        raise ValueError(condition)
    return command


def eval_command(condition: str, seed: int, device: str) -> list[str]:
    return [
        sys.executable,
        "-u",
        "-m",
        "scripts.evaluate_level1",
        "--checkpoint",
        str(run_dir(condition, seed) / "best.pt"),
        "--root",
        str(ROOT / "data/raw/vg150"),
        "--visual-cache",
        str(CACHE),
        "--split",
        "val",
        "--device",
        device,
        "--batch-size",
        "512",
        "--num-workers",
        "0",
        "--output",
        str(report_path(condition, seed)),
    ]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "evaluate"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda", help="Evaluate only; train uses CUDA")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--condition", choices=CONDITIONS, action="append", help="Select one or more conditions")
    parser.add_argument("--seed", type=int, choices=SEEDS, action="append", help="Select one or more seeds")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.max_parallel < 1:
        parser.error("max-parallel must be positive")
    from src.features.visual import load_visual_manifest
    from src.knowledge.export import load_class_features

    conditions = tuple(dict.fromkeys(args.condition)) if args.condition else CONDITIONS
    seeds = tuple(dict.fromkeys(args.seed)) if args.seed else SEEDS
    load_visual_manifest(CACHE, ROOT / "data/raw/vg150", "clip")
    if args.stage == "train" and any(condition in {"kg_on", "kg_off", "edges_off"} for condition in conditions):
        _, valid, manifest = load_class_features(KG)
        if manifest["mode"] != "reviewed" or int(valid.sum()) != 150:
            raise ValueError("Expected a reviewed 150-class KG export")
    tasks = []
    for condition in conditions:
        for seed in seeds:
            label = f"{condition} seed={seed}"
            command = (
                train_command(condition, seed, args.wandb_mode)
                if args.stage == "train"
                else eval_command(condition, seed, args.device)
            )
            log = LOGS / f"{condition}-seed{seed}-{args.stage}.log"
            tasks.append((label, command, log))
    print(f"[PLAN] {args.stage}: {len(tasks)} jobs, max_parallel={args.max_parallel}", flush=True)
    for _, command, log in tasks:
        describe(command, log)
    if args.dry_run:
        return
    existing = [str(log) for _, _, log in tasks if log.exists()]
    if args.stage == "train":
        existing += [str(run_dir(c, s)) for c in conditions for s in seeds if run_dir(c, s).exists()]
    else:
        existing += [
            str(run_dir(c, s) / "best.pt")
            for c in conditions
            for s in seeds
            if not (run_dir(c, s) / "best.pt").is_file()
        ]
        existing += [str(report_path(c, s)) for c in conditions for s in seeds if report_path(c, s).exists()]
        if SUMMARY.exists() and conditions == CONDITIONS and seeds == SEEDS:
            existing.append(str(SUMMARY))
    if existing:
        raise FileExistsError(f"Existing output or missing checkpoint: {existing}")
    LOGS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    run_parallel(tasks, args.max_parallel)
    if args.stage == "evaluate" and conditions == CONDITIONS and seeds == SEEDS:
        command = [
            sys.executable,
            "-m",
            "scripts.summarize_level1",
            *(str(report_path(c, s)) for c in CONDITIONS for s in SEEDS),
            "--output",
            str(SUMMARY),
        ]
        print("[SUMMARY]", shlex.join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
