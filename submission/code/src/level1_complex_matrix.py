"""Launch CLIP + reviewed ConceptNet ComplEx KG-on runs from the user's GPU shell."""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from src.level1_matrix import describe, run_parallel
from src.level1_visual_matrix import CACHE, ROOT, common_args

FEATURES = ROOT / "data/knowledge/c2-conceptnet-complex-toan-v1"
RUNS = ROOT / "runs/level1-clip-complex-v1"
LOGS = ROOT / "logs/level1-clip-complex-v1"
REPORTS = ROOT / "reports/level1"
SEEDS = (42, 123, 3407)
CONDITIONS = ("kg_on", "edges_off")
DEFAULT_CONDITIONS = ("kg_on",)
SUMMARY = REPORTS / "clip-complex-v1-kg_on-3seed-val5000-summary.json"


def run_dir(condition: str, seed: int) -> Path:
    return RUNS / f"clip-complex-{condition}-seed{seed}-v1"


def report_path(condition: str, seed: int) -> Path:
    return REPORTS / f"clip-complex-v1-{condition}-seed{seed}-val5000.json"


def train_command(condition: str, seed: int, wandb_mode: str) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-m",
        "scripts.train_gat_kg",
        "--kg-features",
        str(FEATURES),
        *common_args(seed, wandb_mode),
        "--accumulation-steps",
        "1",
        "--num-heads",
        "4",
        "--num-layers",
        "2",
        "--dropout",
        "0.1",
        "--output-dir",
        str(run_dir(condition, seed)),
    ]
    if condition == "edges_off":
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
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda", help="Evaluate only")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--max-parallel", type=int, default=3)
    parser.add_argument("--condition", choices=CONDITIONS, action="append")
    parser.add_argument("--seed", type=int, choices=SEEDS, action="append")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.max_parallel < 1:
        parser.error("max-parallel must be positive")
    conditions = tuple(dict.fromkeys(args.condition)) if args.condition else DEFAULT_CONDITIONS
    seeds = tuple(dict.fromkeys(args.seed)) if args.seed else SEEDS
    tasks = []
    for condition in conditions:
        for seed in seeds:
            command = (
                train_command(condition, seed, args.wandb_mode)
                if args.stage == "train"
                else eval_command(condition, seed, args.device)
            )
            tasks.append((f"{condition} seed={seed}", command, LOGS / f"{condition}-seed{seed}-{args.stage}.log"))
    print(f"[PLAN] {args.stage}: {len(tasks)} jobs, max_parallel={args.max_parallel}", flush=True)
    for _, command, log in tasks:
        describe(command, log)
    if args.dry_run:
        return
    from src.features.visual import load_visual_manifest
    from src.knowledge.export import load_class_features

    load_visual_manifest(CACHE, ROOT / "data/raw/vg150", "clip")
    if args.stage == "train":
        _, valid, manifest = load_class_features(FEATURES)
        if (
            manifest["mode"] != "reviewed"
            or manifest["kg_source"] != "conceptnet"
            or manifest.get("review_basis") != "human_mixed_direct_and_ai_batch"
            or manifest.get("embedding_model") != "complex"
            or manifest.get("checkpoint_manifest", {}).get("embedding_version") != 1
            or manifest.get("checkpoint_manifest", {}).get("score_function") != "real_hermitian_trilinear_product"
            or manifest.get("dim") != 64
            or int(valid[1:].sum()) != 138
        ):
            raise ValueError("Expected Toan's reviewed ConceptNet ComplEx C.2 bundle with 138 accepted class links")
    existing = [str(log) for _, _, log in tasks if log.exists()]
    if args.stage == "train":
        existing += [str(run_dir(c, s)) for c in conditions for s in seeds if run_dir(c, s).exists()]
    else:
        missing = [
            str(run_dir(c, s) / "best.pt")
            for c in conditions
            for s in seeds
            if not (run_dir(c, s) / "best.pt").is_file()
        ]
        if missing:
            raise FileNotFoundError(f"Missing checkpoints: {missing}")
        existing += [str(report_path(c, s)) for c in conditions for s in seeds if report_path(c, s).exists()]
        if conditions == DEFAULT_CONDITIONS and seeds == SEEDS and SUMMARY.exists():
            existing.append(str(SUMMARY))
    if existing:
        raise FileExistsError(f"Existing outputs or logs: {existing}")
    LOGS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    run_parallel(tasks, args.max_parallel)
    if args.stage == "evaluate" and conditions == DEFAULT_CONDITIONS and seeds == SEEDS:
        command = [
            sys.executable,
            "-m",
            "scripts.summarize_level1",
            *(str(report_path(c, s)) for c in DEFAULT_CONDITIONS for s in SEEDS),
            "--output",
            str(SUMMARY),
        ]
        print("[SUMMARY]", shlex.join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
