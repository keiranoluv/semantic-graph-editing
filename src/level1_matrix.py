"""Launch Level 1 WordNet ablations and evaluate the complete nine-run matrix."""

import argparse
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Direct execution (python src/level1_matrix.py) puts src/, not the repo root,
# on sys.path. The CPU preflight imports src.knowledge from the repo.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
FEATURES = ROOT / "data/knowledge/c2-wordnet-reviewed-v1"
RUNS = ROOT / "runs/level1"
REPORTS = ROOT / "reports/level1"
LOGS = ROOT / "logs/level1"
SEEDS = (42, 123, 3407)
CONDITIONS = ("kg_on", "kg_off", "edges_off")
SUMMARY = REPORTS / "wordnet-reviewed-ablation-3seed-val5000-summary.json"


def run_dir(condition: str, seed: int) -> Path:
    name = f"wordnet-reviewed-seed{seed}-v1" if condition == "kg_on" else f"wordnet-reviewed-{condition}-seed{seed}-v1"
    return RUNS / name


def report_path(condition: str, seed: int) -> Path:
    return REPORTS / f"wordnet-reviewed-{condition}-seed{seed}-val5000.json"


def train_command(condition: str, seed: int) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-m",
        "scripts.train_gat_kg",
        "--kg-features",
        str(FEATURES),
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
        "--accumulation-steps",
        "1",
        "--val-batch-size",
        "128",
        "--max-train-images",
        "57723",
        "--max-val-images",
        "5000",
        "--hidden-dim",
        "256",
        "--embedding-dim",
        "32",
        "--num-heads",
        "4",
        "--num-layers",
        "2",
        "--dropout",
        "0.1",
        "--lr",
        "0.001",
        "--weight-decay",
        "0.01",
        "--negative-ratio",
        "3",
        "--box-policy",
        "clip",
        "--wandb-mode",
        "online",
        "--output-dir",
        str(run_dir(condition, seed)),
    ]
    if condition == "kg_off":
        command.append("--no-knowledge")
    elif condition == "edges_off":
        command.extend(("--message-edges", "none"))
    else:
        raise ValueError(f"Only ablations can be launched here: {condition}")
    return command


def evaluate_command(condition: str, seed: int, device: str) -> list[str]:
    return [
        sys.executable,
        "-u",
        "-m",
        "scripts.evaluate_level1",
        "--checkpoint",
        str(run_dir(condition, seed) / "best.pt"),
        "--split",
        "val",
        "--device",
        device,
        "--batch-size",
        "32",
        "--output",
        str(report_path(condition, seed)),
    ]


def describe(command: list[str], log_path: Path) -> None:
    print(shlex.join(command), f"> {shlex.quote(str(log_path))} 2>&1", flush=True)


def launch(label: str, command: list[str], log_path: Path) -> tuple[str, int]:
    print(f"[START] {label}; log={log_path}", flush=True)
    with log_path.open("x", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False)
    print(f"[DONE] {label}; exit={result.returncode}", flush=True)
    return label, result.returncode


def run_parallel(tasks: list[tuple[str, list[str], Path]], max_parallel: int) -> None:
    remaining = iter(tasks)
    with ThreadPoolExecutor(max_workers=max_parallel) as pool:
        active = {}
        for _ in range(min(max_parallel, len(tasks))):
            task = next(remaining)
            active[pool.submit(launch, *task)] = task[0]
        failed = []
        while active:
            future = next(as_completed(active))
            active.pop(future)
            label, code = future.result()
            if code != 0:
                failed.append(label)
            if not failed:
                task = next(remaining, None)
                if task is not None:
                    active[pool.submit(launch, *task)] = task[0]
    if failed:
        raise RuntimeError(f"Failed runs: {', '.join(failed)}. Remaining jobs were not started; inspect logs.")


def verify_feature_bundle() -> None:
    from src.knowledge.export import load_class_features

    _, valid, manifest = load_class_features(FEATURES)
    if manifest["mode"] != "reviewed" or int(valid.sum()) != 150:
        raise ValueError("A reviewed 150/150 WordNet feature bundle is required")


def verify_completed_runs() -> None:
    import torch

    for condition in CONDITIONS:
        for seed in SEEDS:
            folder = run_dir(condition, seed)
            best, last = folder / "best.pt", folder / "last.pt"
            if not best.is_file() or not last.is_file():
                raise FileNotFoundError(f"Missing best.pt/last.pt: {folder}")
            checkpoint = torch.load(last, map_location="cpu", weights_only=True)
            cfg = checkpoint["config"]
            if (
                cfg.get("model") != "gat_kg"
                or cfg.get("seed") != seed
                or cfg.get("use_knowledge") != (condition != "kg_off")
                or cfg.get("message_edges") != ("none" if condition == "edges_off" else "full")
                or cfg.get("interpretation") != "official_human_selected_representative_synsets_no_per_class_rationale"
            ):
                raise ValueError(f"Wrong configuration for {folder}")
            if checkpoint["epoch"] < cfg["max_epochs"] and checkpoint["bad_epochs"] < cfg["patience"]:
                raise ValueError(f"Training is still incomplete: {folder}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    train = sub.add_parser("train-ablations", help="Launch the six missing KG-off/edges-off runs")
    train.add_argument("--max-parallel", type=int, default=3)
    train.add_argument("--dry-run", action="store_true")
    evaluate = sub.add_parser("evaluate", help="Evaluate all nine completed best checkpoints and aggregate")
    evaluate.add_argument("--max-parallel", type=int, default=2)
    evaluate.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    evaluate.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_parallel < 1:
        parser.error("--max-parallel must be positive")

    if args.stage == "train-ablations":
        tasks = [
            (condition + f" seed={seed}", train_command(condition, seed), LOGS / f"{condition}-seed{seed}-train.log")
            for condition in ("kg_off", "edges_off")
            for seed in SEEDS
        ]
        verify_feature_bundle()
        if not args.dry_run:
            existing = [
                str(run_dir(condition, seed))
                for condition in ("kg_off", "edges_off")
                for seed in SEEDS
                if run_dir(condition, seed).exists()
            ]
            existing += [str(log) for _, _, log in tasks if log.exists()]
            if existing:
                raise FileExistsError(f"A run or log already exists; refusing to overwrite: {existing}")
    else:
        tasks = [
            (
                condition + f" seed={seed}",
                evaluate_command(condition, seed, args.device),
                LOGS / f"{condition}-seed{seed}-eval.log",
            )
            for condition in CONDITIONS
            for seed in SEEDS
        ]
        if not args.dry_run:
            verify_completed_runs()
            existing = [
                str(report_path(condition, seed))
                for condition in CONDITIONS
                for seed in SEEDS
                if report_path(condition, seed).exists()
            ]
            existing += [str(log) for _, _, log in tasks if log.exists()]
            if SUMMARY.exists():
                existing.append(str(SUMMARY))
            if existing:
                raise FileExistsError(f"A report, summary or log already exists; refusing to overwrite: {existing}")

    print(f"[PLAN] stage={args.stage}; jobs={len(tasks)}; max_parallel={args.max_parallel}", flush=True)
    for _, command, log in tasks:
        describe(command, log)
    if args.dry_run:
        return
    LOGS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    run_parallel(tasks, args.max_parallel)
    if args.stage == "evaluate":
        command = [
            sys.executable,
            "-m",
            "scripts.summarize_level1",
            *(str(report_path(condition, seed)) for condition in CONDITIONS for seed in SEEDS),
            "--output",
            str(SUMMARY),
        ]
        print("[SUMMARY]", shlex.join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
