"""Evaluate 12 loss and 15 fusion checkpoints on full VG150 test, sequentially."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path

from tools import level1_reference_test27_v1 as runner

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/level1/reference-loss-fusion-test27-v1"
SOURCES = (
    ("loss", "reference-v1-pair", "86c4e1c915a5da9b53ca3fb1e0dc2f9ad6c998f81cb46fc1e77dc613b134905f"),
    ("fusion", "reference-v1-fusion", "39e371ee2762521a4c3dea570d0d0526c565d7f47ac2c0fb1ff7de97a8c6382d"),
)
CONDITIONS = {
    "loss": {"ce_off", "ce_on", "adjusted_off", "adjusted_on"},
    "fusion": {"baseline", "pair_real", "pair_shuffled", "visual_real", "visual_shuffled"},
}


def build_plan():
    plan = {"purpose": "fixed_checkpoint_loss_fusion_test_no_reselection", "entries": [], "sources": []}
    for group, directory, digest in SOURCES:
        path = ROOT / "reports/level1" / directory / "frozen-checkpoints.json"
        if runner.sha256(path) != digest:
            raise ValueError(f"Frozen source plan changed: {path}")
        source = json.loads(path.read_text())
        expected = {(c, s) for c in CONDITIONS[group] for s in (42, 123, 3407)}
        actual = {(e["condition"], e["seed"]) for e in source["entries"]}
        if actual != expected or len(source["entries"]) != len(expected):
            raise ValueError("Unexpected conditions or seeds")
        for key in ("data_sha256", "visual_cache_manifest_sha256"):
            if key in plan and plan[key] != source[key]:
                raise ValueError(f"Input mismatch: {key}")
            plan[key] = source[key]
        plan["sources"].append({"path": str(path.relative_to(ROOT)), "sha256": digest})
        for entry in source["entries"]:
            plan["entries"].append({**entry, "condition": group + "_" + entry["condition"]})
    hashes = [e["checkpoint_sha256"] for e in plan["entries"]]
    original = json.loads(runner.PLAN.read_text())
    if len(set(hashes)) != 27 or set(hashes) & {e["checkpoint_sha256"] for e in original["entries"]}:
        raise ValueError("Duplicate checkpoint or overlap with original 27 runs")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("batch-size must be positive")
    # Reuse the verified runner without modifying its historical source on disk.
    if runner.sha256(ROOT / "tools/level1_reference_test27_v1.py") != (
        "38cca5092687e2139f9e758ef54a5e6976988a4c8567ddf17a25e735ebf24298"
    ):
        raise ValueError("Shared runner changed")
    plan = build_plan()
    encoded = json.dumps(plan, indent=2) + "\n"
    runner.OUTPUT = OUTPUT
    runner.PLAN_HASH = hashlib.sha256(encoded.encode()).hexdigest()
    frozen = OUTPUT / "frozen-checkpoints.json"
    if frozen.exists() and frozen.read_text() != encoded:
        raise ValueError("Output plan differs; refusing overwrite")
    if args.dry_run:
        runner.execute(plan, args.batch_size, True)
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / ".launcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not frozen.exists():
            with frozen.open("x") as stream:
                stream.write(encoded)
        runner.execute(plan, args.batch_size, False)


if __name__ == "__main__":
    main()
