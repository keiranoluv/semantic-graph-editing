"""Combine full PredCls evaluation reports across matched seeds and protocols."""

import argparse
import json
import math
import statistics
from pathlib import Path

METRICS = ("R@50", "R@100", "mR@50", "mR@100")


def condition(report):
    cfg = report["config"]
    if cfg["model"] == "gat_kg":
        source = report["config"].get("kg_manifest_sha256", "unknown")[:12]
        kg_source = "conceptnet" if "frozen_conceptnet" in cfg.get("features", "") else "wordnet"
        return f"gat_kg:{kg_source}:{source}:kg_{int(cfg['use_knowledge'])}:edges_{cfg['message_edges']}"
    return cfg["model"]


def summarize(paths, *, expected_seeds=(42, 123, 3407), allow_subset=False):
    reports = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    if not reports:
        raise ValueError("No evaluation reports supplied")
    if len(set(expected_seeds)) != len(expected_seeds) or len(expected_seeds) < 3:
        raise ValueError("Expected at least three distinct seeds")
    reference = reports[0]
    keys = ("split", "protocol", "data_sha256", "images_evaluated", "split_images")
    if reference.get("split") != "val":
        raise ValueError("Only validation reports are summarized before final test release")
    if not allow_subset and not reference.get("full_split"):
        raise ValueError("Subset evaluation is technical only; full split required")
    ref_support = {key: value["support"] for key, value in reference["per_predicate"].items()}
    ref_visual = reference["config"].get("visual_cache_manifest_sha256")
    matched_config_keys = (
        "geometry_dim",
        "box_policy",
        "train_limit",
        "val_limit",
        "negative_ratio",
        "batch_size",
        "val_batch_size",
        "patience",
        "min_delta",
        "lr",
        "weight_decay",
        "hidden_dim",
        "embedding_dim",
        "num_workers",
        "deterministic",
        "code_sha256",
    )
    ref_matched_config = {key: reference["config"].get(key) for key in matched_config_keys}
    groups = {}
    for report in reports:
        if any(report.get(k) != reference.get(k) for k in keys):
            raise ValueError("Reports differ in split, protocol, data hashes or image count")
        if not allow_subset and not report.get("full_split"):
            raise ValueError("Subset evaluation is technical only; full split required")
        support = {key: value["support"] for key, value in report["per_predicate"].items()}
        if support != ref_support:
            raise ValueError("Per-predicate support differs; evaluated images/protocol may differ")
        config = report["config"]
        if config.get("visual_cache_manifest_sha256") != ref_visual:
            raise ValueError("Reports use different frozen visual caches")
        if ref_visual is not None and {key: config.get(key) for key in matched_config_keys} != ref_matched_config:
            raise ValueError("CLIP reports differ in shared input or training protocol")
        if config.get("seed") != report.get("seed"):
            raise ValueError("Checkpoint config seed differs from evaluation report")
        if config.get("val_limit") is not None and config["val_limit"] != report["images_evaluated"]:
            raise ValueError("Checkpoint selection and evaluation used different validation sizes")
        if any(not isinstance(report.get(k), (int, float)) or not math.isfinite(report[k]) for k in METRICS):
            raise ValueError("Missing or invalid metric")
        groups.setdefault(condition(report), []).append(report)
    output = []
    for name, rows in sorted(groups.items()):
        seeds = [r["seed"] for r in rows]
        if sorted(seeds) != sorted(expected_seeds):
            raise ValueError(f"{name}: expected seeds {list(expected_seeds)}, got {seeds}")
        cfg_ref = {k: v for k, v in rows[0]["config"].items() if k != "seed"}
        if any({k: v for k, v in r["config"].items() if k != "seed"} != cfg_ref for r in rows[1:]):
            raise ValueError(f"{name}: training configs differ beyond seed")
        if len({r["checkpoint_sha256"] for r in rows}) != len(rows):
            raise ValueError(f"{name}: duplicate checkpoint")
        output.append(
            {
                "condition": name,
                "seeds": sorted(seeds),
                "reports": [r["checkpoint"] for r in sorted(rows, key=lambda r: r["seed"])],
                "metrics": {
                    metric: {
                        "mean": statistics.mean(r[metric] for r in rows),
                        "sample_std": statistics.stdev(r[metric] for r in rows),
                    }
                    for metric in METRICS
                },
                "max_epochs_recorded": rows[0]["config"].get("max_epochs"),
            }
        )
    return {
        "schema_version": 1,
        "split": "val",
        "images_evaluated": reference["images_evaluated"],
        "full_split": reference["full_split"],
        "interpretation": "technical_subset_only" if not reference["full_split"] else "validation_not_final_test",
        "data_sha256": reference["data_sha256"],
        "protocol": reference["protocol"],
        "visual_cache_manifest_sha256": ref_visual,
        "metrics_unit": "fraction_0_to_1",
        "std_definition": "sample_standard_deviation_ddof_1",
        "groups": output,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reports", nargs="+", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--allow-subset", action="store_true", help="Technical aggregation only")
    args = p.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    summary = summarize(args.reports, allow_subset=args.allow_subset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(summary, indent=2) + "\n")
    for group in summary["groups"]:
        print(group["condition"], {key: value for key, value in group["metrics"].items()})


if __name__ == "__main__":
    main()
