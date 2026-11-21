"""Freeze and evaluate the 27 CLIP PredCls checkpoints on held-out VG150 test.

This launcher does not change the training or evaluation implementation. GPU
evaluation is run by the project owner after committing the frozen plan.
"""

import argparse
import hashlib
import json
import math
import shlex
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.level1_matrix import run_parallel  # noqa: E402

RAW = ROOT / "data/raw/vg150"
CACHE = ROOT / "data/cache/vg150-clip-vit-b32-v1"
VAL_REPORTS = ROOT / "reports/level1"
TEST_REPORTS = VAL_REPORTS / "test-v1"
LOGS = ROOT / "logs/level1-clip-test-v1"
PLAN = VAL_REPORTS / "clip-all-27run-test-plan.json"
SUMMARY = VAL_REPORTS / "clip-all-27run-test26446-summary.json"
SEEDS = (42, 123, 3407)
METRICS = ("R@50", "mR@50", "R@100", "mR@100")
CONDITIONS = {
    "mlp_semantic": "clip-v1-mlp_semantic-seed{seed}-val5000.json",
    "gat": "clip-v1-gat-seed{seed}-val5000.json",
    "kg_off": "clip-v1-kg_off-seed{seed}-val5000.json",
    "wordnet_kg_on": "clip-v1-kg_on-seed{seed}-val5000.json",
    "wordnet_edges_off": "clip-v1-edges_off-seed{seed}-val5000.json",
    "conceptnet_transe_kg_on": "clip-conceptnet-v1-kg_on-seed{seed}-val5000.json",
    "conceptnet_transe_edges_off": "clip-conceptnet-v1-edges_off-seed{seed}-val5000.json",
    "conceptnet_rotate_kg_on": "clip-rotate-v2-kg_on-seed{seed}-val5000.json",
    "conceptnet_complex_kg_on": "clip-complex-v1-kg_on-seed{seed}-val5000.json",
}
SOURCE_FILES = (
    "tools/level1_test_matrix.py",
    "scripts/evaluate_level1.py",
    "scripts/evaluate_mlp.py",
    "scripts/pilot_mlp.py",
    "scripts/train_mlp.py",
    "src/datasets/batching.py",
    "src/datasets/hdf5_session.py",
    "src/datasets/schemas.py",
    "src/datasets/vg150.py",
    "src/evaluation/batched.py",
    "src/evaluation/gat.py",
    "src/evaluation/scene_graph.py",
    "src/features/geometry.py",
    "src/features/visual.py",
    "src/graph/builder.py",
    "src/graph/gnn_batching.py",
    "src/models/gat.py",
    "src/models/gat_kg.py",
    "src/models/mlp_semantic.py",
)
SHARED_CONFIG = (
    "train_limit",
    "val_limit",
    "batch_size",
    "val_batch_size",
    "negative_ratio",
    "geometry_dim",
    "box_policy",
    "hidden_dim",
    "embedding_dim",
    "patience",
    "min_delta",
    "lr",
    "weight_decay",
    "deterministic",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def freeze_plan() -> None:
    if PLAN.exists():
        raise FileExistsError(f"Frozen test plan already exists: {PLAN}")
    if TEST_REPORTS.exists() and any(TEST_REPORTS.glob("*-test26446.json")):
        raise FileExistsError(f"Test reports already exist before plan freeze: {TEST_REPORTS}")
    splits = read_json(RAW / "splits.json")
    test_images = len(splits["test"])
    if test_images != 26446:
        raise ValueError(f"Unexpected test size: {test_images}")

    entries = []
    reference = None
    for condition, filename in CONDITIONS.items():
        for seed in SEEDS:
            val_path = VAL_REPORTS / filename.format(seed=seed)
            report = read_json(val_path)
            cfg = report["config"]
            if report["split"] != "val" or not report["full_split"] or report["images_evaluated"] != 5000:
                raise ValueError(f"Incomplete validation report: {val_path}")
            if report["seed"] != seed or cfg["seed"] != seed:
                raise ValueError(f"Seed mismatch: {val_path}")
            expected_model = (
                "mlp_semantic" if condition == "mlp_semantic" else "gat" if condition == "gat" else "gat_kg"
            )
            if cfg["model"] != expected_model:
                raise ValueError(f"Wrong model for condition {condition}: {val_path}")
            if cfg["model"] == "gat_kg":
                expected_source = "conceptnet" if condition.startswith("conceptnet_") else "wordnet"
                expected_edges = "none" if condition.endswith("edges_off") else "full"
                if (
                    expected_source not in cfg["features"]
                    or cfg["use_knowledge"] != (condition != "kg_off")
                    or cfg["message_edges"] != expected_edges
                ):
                    raise ValueError(f"Wrong KG configuration for condition {condition}: {val_path}")
            checkpoint = Path(report["checkpoint"]).resolve()
            relative(checkpoint)  # Refuse paths outside this checkout.
            if sha256(checkpoint) != report["checkpoint_sha256"]:
                raise ValueError(f"Validation report/checkpoint hash mismatch: {val_path}")
            identity = (
                report["data_sha256"],
                cfg["visual_cache_manifest_sha256"],
                report["protocol"],
                tuple((key, cfg.get(key)) for key in SHARED_CONFIG),
            )
            if reference is None:
                reference = identity
            elif identity != reference:
                raise ValueError(f"Input or training protocol differs: {val_path}")
            entries.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "validation_report": relative(val_path),
                    "validation_report_sha256": sha256(val_path),
                    "checkpoint": relative(checkpoint),
                    "checkpoint_sha256": report["checkpoint_sha256"],
                    "checkpoint_epoch": report["checkpoint_epoch"],
                    "validation_mR@50": report["mR@50"],
                }
            )

    plan = {
        "schema_version": 1,
        "purpose": "one_pass_held_out_test_after_validation_model_selection",
        "split": "test",
        "test_images": test_images,
        "protocol": reference[2],
        "data_sha256": reference[0],
        "visual_cache_manifest_sha256": reference[1],
        "shared_training_config": dict(reference[3]),
        "evaluation": {"batch_size": 512, "num_workers": 0, "device": "cuda"},
        "source_sha256": {name: sha256(ROOT / name) for name in SOURCE_FILES},
        "conditions": list(CONDITIONS),
        "seeds": list(SEEDS),
        "entries": entries,
    }
    PLAN.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(f"[FROZEN] {len(entries)} checkpoints; test images={test_images}; plan={PLAN}")


def load_plan() -> dict:
    plan = read_json(PLAN)
    if (
        plan["schema_version"] != 1
        or plan["split"] != "test"
        or plan["test_images"] != 26446
        or plan["conditions"] != list(CONDITIONS)
        or plan["seeds"] != list(SEEDS)
        or len(plan["entries"]) != 27
    ):
        raise ValueError("Unexpected frozen test plan")
    expected_entries = {(condition, seed) for condition in CONDITIONS for seed in SEEDS}
    actual_entries = {(entry["condition"], entry["seed"]) for entry in plan["entries"]}
    if actual_entries != expected_entries:
        raise ValueError("Frozen test plan has missing or duplicate condition/seed entries")
    for name, expected in plan["source_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Evaluation source changed after plan freeze: {name}")
    for entry in plan["entries"]:
        checkpoint = ROOT / entry["checkpoint"]
        val_report = ROOT / entry["validation_report"]
        if sha256(checkpoint) != entry["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint changed after plan freeze: {checkpoint}")
        if sha256(val_report) != entry["validation_report_sha256"]:
            raise ValueError(f"Validation report changed after plan freeze: {val_report}")
    if len(read_json(RAW / "splits.json")["test"]) != plan["test_images"]:
        raise ValueError("Test split size changed after plan freeze")
    return plan


def report_path(entry: dict) -> Path:
    return TEST_REPORTS / f"{entry['condition']}-seed{entry['seed']}-test26446.json"


def log_path(entry: dict) -> Path:
    return LOGS / f"{entry['condition']}-seed{entry['seed']}-test.log"


def verify_test_report(entry: dict, plan: dict) -> dict:
    path = report_path(entry)
    report = read_json(path)
    cfg = report["config"]
    required = (
        report["split"] == "test",
        report["full_split"] is True,
        report["images_evaluated"] == plan["test_images"],
        report["split_images"] == plan["test_images"],
        report["seed"] == entry["seed"] == cfg["seed"],
        report["checkpoint_epoch"] == entry["checkpoint_epoch"],
        report["checkpoint_sha256"] == entry["checkpoint_sha256"],
        report["data_sha256"] == plan["data_sha256"],
        report["protocol"] == plan["protocol"],
        cfg["visual_cache_manifest_sha256"] == plan["visual_cache_manifest_sha256"],
        relative(Path(report["checkpoint"])) == entry["checkpoint"],
    )
    if not all(required) or any(
        not isinstance(report.get(m), (int, float)) or not math.isfinite(report[m]) for m in METRICS
    ):
        raise ValueError(f"Invalid held-out test report: {path}")
    return report


def summarize(plan: dict) -> None:
    groups = []
    support_reference = None
    for condition in CONDITIONS:
        entries = [entry for entry in plan["entries"] if entry["condition"] == condition]
        reports = [verify_test_report(entry, plan) for entry in entries]
        for report in reports:
            support = {key: value["support"] for key, value in report["per_predicate"].items()}
            if support_reference is None:
                support_reference = support
            elif support != support_reference:
                raise ValueError("Per-predicate test support differs across reports")
        groups.append(
            {
                "condition": condition,
                "seeds": list(SEEDS),
                "reports": [relative(report_path(entry)) for entry in entries],
                "metrics": {
                    metric: {
                        "mean": statistics.mean(report[metric] for report in reports),
                        "sample_std": statistics.stdev(report[metric] for report in reports),
                    }
                    for metric in METRICS
                },
            }
        )
    result = {
        "schema_version": 1,
        "split": "test",
        "images_evaluated": plan["test_images"],
        "full_split": True,
        "interpretation": "held_out_test_no_tuning",
        "plan_sha256": sha256(PLAN),
        "data_sha256": plan["data_sha256"],
        "protocol": plan["protocol"],
        "visual_cache_manifest_sha256": plan["visual_cache_manifest_sha256"],
        "metrics_unit": "fraction_0_to_1",
        "std_definition": "sample_standard_deviation_ddof_1",
        "num_predicates_with_gt": len(support_reference),
        "groups": groups,
    }
    if SUMMARY.exists():
        if read_json(SUMMARY) != result:
            raise FileExistsError(f"Existing summary differs from verified reports: {SUMMARY}")
    else:
        SUMMARY.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"[SUMMARY] {SUMMARY}")
    for group in groups:
        print(group["condition"], {m: round(group["metrics"][m]["mean"] * 100, 4) for m in METRICS})


def run(plan: dict, device: str, max_parallel: int, dry_run: bool) -> None:
    if device != plan["evaluation"]["device"]:
        raise ValueError(f"Frozen plan requires device={plan['evaluation']['device']}")
    tasks = []
    complete = 0
    for entry in plan["entries"]:
        output = report_path(entry)
        log = log_path(entry)
        if output.exists():
            verify_test_report(entry, plan)
            complete += 1
            continue
        if log.exists():
            raise FileExistsError(f"Log exists without a valid report; inspect and archive before retry: {log}")
        command = [
            sys.executable,
            "-u",
            "-m",
            "scripts.evaluate_level1",
            "--checkpoint",
            str(ROOT / entry["checkpoint"]),
            "--root",
            str(RAW),
            "--visual-cache",
            str(CACHE),
            "--split",
            "test",
            "--allow-test",
            "--device",
            device,
            "--batch-size",
            str(plan["evaluation"]["batch_size"]),
            "--num-workers",
            str(plan["evaluation"]["num_workers"]),
            "--output",
            str(output),
        ]
        tasks.append((f"{entry['condition']} seed={entry['seed']}", command, log))
    print(
        f"[PLAN] complete={complete}; remaining={len(tasks)}; max_parallel={max_parallel}; test_images={plan['test_images']}"
    )
    if dry_run:
        for label, command, log in tasks:
            print(label, shlex.join(command), f"> {shlex.quote(str(log))} 2>&1")
        return
    if tasks:
        TEST_REPORTS.mkdir(parents=True, exist_ok=True)
        LOGS.mkdir(parents=True, exist_ok=True)
        run_parallel(tasks, max_parallel)
    summarize(plan)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run", "summarize"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-parallel", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_parallel < 1:
        parser.error("--max-parallel must be positive")
    if args.stage == "freeze":
        freeze_plan()
        return
    plan = load_plan()
    if args.stage == "summarize":
        summarize(plan)
    else:
        run(plan, args.device, args.max_parallel, args.dry_run)


if __name__ == "__main__":
    main()
