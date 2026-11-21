"""Sequential reference-v1 test evaluation of the original 27 frozen checkpoints."""
import argparse
import fcntl
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

from src.evaluation.reference_predcls import PROTOCOL
from src.knowledge.transe.core import sha256

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "reports/level1/clip-all-27run-test-plan.json"
PLAN_HASH = "2220ff667068691a4cae6334a0bf3f529e75b3e9e652f4f46cbab58d986c8add"
OUTPUT = ROOT / "reports/level1/reference-test27-v1"
METRICS = ("R@50", "R@100", "mR@50", "mR@100")


def validate(row, entry, plan):
    expected = dict(
        checkpoint_sha256=entry["checkpoint_sha256"],
        seed=entry["seed"],
        protocol=PROTOCOL,
        split="test",
        full_split=True,
        images_evaluated=26446,
        data_sha256=plan["data_sha256"],
    )
    if any(row.get(k) != v for k, v in expected.items()):
        raise ValueError("Report incompatible with frozen test plan")
    if row["config"]["visual_cache_manifest_sha256"] != plan["visual_cache_manifest_sha256"]:
        raise ValueError("Visual cache mismatch")
    hashes = row.get("evaluator_sha256", {})
    if not hashes or any(sha256(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError("Evaluator provenance missing or changed")
    if any(not math.isfinite(row[k]) or not 0 <= row[k] <= 1 for k in METRICS):
        raise ValueError("Invalid metrics")


def execute(plan, batch, dry_run):
    entries = plan["entries"]
    if len(entries) != 27 or len({(e["condition"], e["seed"]) for e in entries}) != 27:
        raise ValueError("Expected 27 distinct runs")
    jobs = []
    # Preflight every checkpoint and existing report before launching any inference.
    for entry in entries:
        cp = ROOT / entry["checkpoint"]
        if sha256(cp) != entry["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint changed: {cp}")
        path = OUTPUT / f"{entry['condition']}-seed{entry['seed']}-test26446.json"
        row = json.loads(path.read_text()) if path.exists() else None
        if row is not None:
            validate(row, entry, plan)
        jobs.append((entry, cp, path, row))
    print(f"Preflight PASS: 27 checkpoints; reuse={sum(r is not None for _, _, _, r in jobs)}", flush=True)
    reports, groups = [], {}
    for entry, cp, path, row in jobs:
        command = [
            sys.executable,
            "-m",
            "scripts.evaluate_level1",
            "--checkpoint",
            str(cp),
            "--protocol",
            "reference-v1",
            "--split",
            "test",
            "--allow-test",
            "--device",
            "cuda",
            "--batch-size",
            str(batch),
            "--output",
            str(path),
        ]
        print(("SKIP " if row is not None else "RUN ") + path.name, flush=True)
        if dry_run:
            if row is None:
                print(" ".join(command))
            continue
        if row is None:
            subprocess.run(command, cwd=ROOT, check=True)
            row = json.loads(path.read_text())
            validate(row, entry, plan)
        groups.setdefault(entry["condition"], []).append(row)
        reports.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path)})
    if dry_run:
        return
    summary = []
    for condition, rows in groups.items():
        if sorted(r["seed"] for r in rows) != [42, 123, 3407]:
            raise ValueError("Incomplete seed group")
        summary.append(
            {
                "condition": condition,
                "metrics": {
                    k: {"mean": statistics.mean(r[k] for r in rows), "sample_std": statistics.stdev(r[k] for r in rows)}
                    for k in METRICS
                },
            }
        )
    result = {
        "split": "test",
        "protocol": PROTOCOL,
        "images_evaluated": 26446,
        "plan_sha256": PLAN_HASH,
        "metrics_unit": "fraction_0_to_1",
        "interpretation": "fixed_checkpoint_rescore_previously_viewed_test_no_reselection",
        "reports": reports,
        "groups": summary,
    }
    path = OUTPUT / "summary.json"
    encoded = json.dumps(result, indent=2) + "\n"
    if path.exists():
        if path.read_text() != encoded:
            raise ValueError("Existing summary differs; refusing overwrite")
    else:
        with path.open("x") as stream:
            stream.write(encoded)
    print(f"Completed 27/27: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("batch-size must be positive")
    if sha256(PLAN) != PLAN_HASH:
        raise ValueError("Original frozen plan changed")
    plan = json.loads(PLAN.read_text())
    if args.dry_run:
        execute(plan, args.batch_size, True)
    else:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        with (OUTPUT / ".launcher.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            execute(plan, args.batch_size, False)


if __name__ == "__main__":
    main()
