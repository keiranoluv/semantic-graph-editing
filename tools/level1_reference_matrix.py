"""Rescore frozen main, pair/loss, or eligible fusion checkpoints on validation."""

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

from src.evaluation.reference_predcls import NG_PROTOCOL, PROTOCOL
from src.knowledge.transe.core import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--matrix", choices=("main", "pair", "fusion"), default="main")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--protocol", choices=("reference-v1", "reference-ng-v1"), default="reference-v1")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("batch-size must be positive")
    if args.output_dir is None:
        args.output_dir = Path("reports/level1/" + args.protocol + ("" if args.matrix == "main" else "-" + args.matrix))
    protocol_id = NG_PROTOCOL if args.protocol == "reference-ng-v1" else PROTOCOL
    if args.matrix == "main":
        plan_path = Path("reports/level1/clip-all-27run-test-plan.json")
        plan = json.loads(plan_path.read_text())
    else:
        conditions = (
            ["ce_off", "ce_on", "adjusted_off", "adjusted_on"]
            if args.matrix == "pair"
            else ["baseline", "pair_real", "pair_shuffled", "visual_real", "visual_shuffled"]
        )
        prefix = "clip-pair-kg-v1" if args.matrix == "pair" else "clip-fusion-v1"
        entries = []
        reference = None
        for condition in conditions:
            for seed in (42, 123, 3407):
                source = Path(f"reports/level1/{prefix}-{condition}-seed{seed}-val5000.json")
                old = json.loads(source.read_text())
                if reference is None:
                    reference = old
                if old["data_sha256"] != reference["data_sha256"]:
                    raise ValueError("Legacy inputs differ")
                entries.append(
                    {
                        "condition": condition,
                        "seed": seed,
                        "checkpoint": old["checkpoint"],
                        "checkpoint_sha256": old["checkpoint_sha256"],
                        "legacy_report": str(source),
                        "legacy_report_sha256": sha256(source),
                    }
                )
        plan = {
            "purpose": "fixed_checkpoint_validation_rescore",
            "entries": entries,
            "data_sha256": reference["data_sha256"],
            "visual_cache_manifest_sha256": reference["config"]["visual_cache_manifest_sha256"],
            "excluded_conditions": ["relation_real", "relation_shuffled"] if args.matrix == "fusion" else [],
            "exclusion_reason": "TransE training L1 versus relation-score export L2; audited before rescoring"
            if args.matrix == "fusion"
            else None,
        }
        plan_path = args.output_dir / "frozen-checkpoints.json"
        encoded_plan = json.dumps(plan, indent=2) + "\n"
        if plan_path.exists() and plan_path.read_text() != encoded_plan:
            raise ValueError("Frozen rescore plan changed")
        if not args.dry_run and not plan_path.exists():
            args.output_dir.mkdir(parents=True, exist_ok=True)
            with plan_path.open("x") as stream:
                stream.write(encoded_plan)
    groups = {}
    reports = []
    for i, entry in enumerate(plan["entries"], 1):
        cp = Path(entry["checkpoint"])
        if sha256(cp) != entry["checkpoint_sha256"]:
            raise ValueError(f"Frozen checkpoint changed: {cp}")
        output = args.output_dir / f"{entry['condition']}-seed{entry['seed']}-val5000.json"
        cmd = [
            sys.executable,
            "-m",
            "scripts.evaluate_level1",
            "--checkpoint",
            str(cp),
            "--protocol",
            args.protocol,
            "--split",
            "val",
            "--device",
            args.device,
            "--batch-size",
            str(args.batch_size),
            "--output",
            str(output),
        ]
        if args.dry_run:
            print(" ".join(cmd), flush=True)
            continue
        print(f"[{i}/{len(plan['entries'])}] {entry['condition']} seed {entry['seed']}", flush=True)
        if not output.exists():
            subprocess.run(cmd, check=True)
        row = json.loads(output.read_text())
        if not (
            row["checkpoint_sha256"] == entry["checkpoint_sha256"]
            and row["protocol"] == protocol_id
            and row["split"] == "val"
            and row["full_split"]
            and row["images_evaluated"] == 5000
            and row["data_sha256"] == plan["data_sha256"]
            and row["seed"] == entry["seed"]
            and row["config"]["visual_cache_manifest_sha256"] == plan["visual_cache_manifest_sha256"]
        ):
            raise ValueError(f"Incompatible existing report: {output}")
        if any(sha256(Path(path)) != digest for path, digest in row["evaluator_sha256"].items()):
            raise ValueError(f"Report was produced by different evaluator code: {output}")
        groups.setdefault(entry["condition"], []).append(row)
        reports.append({"path": str(output), "sha256": sha256(output)})
    if args.dry_run:
        return
    prefix = "ng-" if args.protocol == "reference-ng-v1" else ""
    metrics = tuple(prefix + key for key in ("R@50", "R@100", "mR@50", "mR@100"))
    summary = []
    for condition, rows in groups.items():
        if sorted(row["seed"] for row in rows) != [42, 123, 3407]:
            raise ValueError("Expected three distinct seeds per condition")
        summary.append(
            {
                "condition": condition,
                "seeds": [42, 123, 3407],
                "metrics": {
                    key: {
                        "mean": statistics.mean(r[key] for r in rows),
                        "sample_std": statistics.stdev(r[key] for r in rows),
                    }
                    for key in metrics
                },
            }
        )
    result = {
        "protocol": protocol_id,
        "split": "val",
        "images_evaluated": 5000,
        "full_split": True,
        "interpretation": "fixed_legacy_selected_checkpoints_rescored_not_reselected",
        "plan_sha256": sha256(plan_path),
        "metrics_unit": "fraction_0_to_1",
        "std_definition": "sample_standard_deviation_ddof_1",
        "reports": reports,
        "groups": summary,
    }
    summary_path = args.output_dir / f"{len(plan['entries'])}run-val5000-summary.json"
    encoded = json.dumps(result, indent=2) + "\n"
    if summary_path.exists():
        if summary_path.read_text() != encoded:
            raise FileExistsError(summary_path)
    else:
        with summary_path.open("x") as stream:
            stream.write(encoded)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
