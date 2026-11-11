"""Verify and compare the frozen 54-run graph-constrained and unconstrained reports."""

import argparse
import json
import math
import statistics
from pathlib import Path

from src.evaluation.reference_predcls import NG_PROTOCOL, PROTOCOL
from src.knowledge.transe.core import sha256


def checked_reports(path, protocol):
    summary = json.loads(path.read_text())
    if summary["protocol"] != protocol:
        raise ValueError(f"Wrong protocol: {path}")
    rows = {}
    for item in summary["reports"]:
        source = Path(item["path"])
        if sha256(source) != item["sha256"]:
            raise ValueError(f"Changed report: {source}")
        row = json.loads(source.read_text())
        if (
            row["protocol"] != protocol
            or row["split"] != "val"
            or not row["full_split"]
            or row["images_evaluated"] != 5000
        ):
            raise ValueError(f"Incompatible report: {source}")
        if sha256(Path(row["checkpoint"])) != row["checkpoint_sha256"]:
            raise ValueError(f"Changed checkpoint: {source}")
        if protocol == NG_PROTOCOL:
            for filename, digest in row["evaluator_sha256"].items():
                if sha256(Path(filename)) != digest:
                    raise ValueError(f"Changed NG evaluator: {filename}")
        if source.name in rows:
            raise ValueError(f"Duplicate report: {source}")
        rows[source.name] = row
    prefix = "ng-" if protocol == NG_PROTOCOL else ""
    for group in summary["groups"]:
        group_rows = [rows[f"{group['condition']}-seed{seed}-val5000.json"] for seed in group["seeds"]]
        if sorted(r["seed"] for r in group_rows) != [42, 123, 3407]:
            raise ValueError("Expected three seeds")
        for metric in ("R@50", "mR@50", "R@100", "mR@100"):
            values = [r[prefix + metric] for r in group_rows]
            expected = {"mean": statistics.mean(values), "sample_std": statistics.stdev(values)}
            if any(
                not math.isclose(group["metrics"][prefix + metric][k], v, abs_tol=1e-12) for k, v in expected.items()
            ):
                raise ValueError(f"Summary mismatch: {path}")
    return summary, rows


def compare_reports(gc, ng):
    for key in ("checkpoint_sha256", "checkpoint_epoch", "seed", "data_sha256", "config", "split_images"):
        if gc[key] != ng[key]:
            raise ValueError(f"Unmatched reports: {key}")
    for key in ("support", "images_with_gt"):
        if {c: r[key] for c, r in gc["per_predicate"].items()} != {c: r[key] for c, r in ng["per_predicate"].items()}:
            raise ValueError(f"Different per-predicate {key}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("reports/level1/reference-comparison"))
    args = p.parse_args()
    result = {
        "status": "PASS",
        "scope": "54_frozen_checkpoints_full_validation_5000_images_three_seeds_no_reselection",
        "protocols": {"graph_constraint": PROTOCOL, "no_graph_constraint": NG_PROTOCOL},
        "metrics_unit": "fraction_0_to_1",
        "delta_unit": "percentage_points",
        "std_definition": "sample_standard_deviation_ddof_1",
        "checks": [
            "report_hashes",
            "checkpoint_hashes",
            "same_inputs_and_config",
            "same_class_support",
            "three_seed_means_and_sample_stds",
            "current_ng_evaluator_hashes",
        ],
        "historical_gc_evaluator": "retained_source_hashes_in_original_reports; current_both_modes_pass_pinned_differential_check",
        "matrices": {},
        "paired_effects": [],
    }
    text = [
        "# So sánh graph constraint và no graph constraint — Mức 1",
        "",
        "54 checkpoint cố định, cùng 5.000 ảnh validation, seed 42/123/3407. "
        "Đơn vị %, mean ± sample std. Không train/chọn lại checkpoint; metric chọn checkpoint vẫn là legacy.",
        "",
        "GC: mỗi cặp có hướng giữ một predicate. NG: tất cả predicate foreground tham gia top-K toàn ảnh. "
        "Matching và cách trung bình recall giống nhau. Hai quan hệ cùng cặp chiếm hai vị trí; NG không bảo đảm tăng recall.",
    ]
    all_rows = {}
    for family, count in (("main", 27), ("pair", 12), ("fusion", 15)):
        suffix = "" if family == "main" else "-" + family
        gc_path = Path(f"reports/level1/reference-v1{suffix}/{count}run-val5000-summary.json")
        ng_path = Path(f"reports/level1/reference-ng-v1{suffix}/{count}run-val5000-summary.json")
        gc_summary, gc_rows = checked_reports(gc_path, PROTOCOL)
        ng_summary, ng_rows = checked_reports(ng_path, NG_PROTOCOL)
        if gc_rows.keys() != ng_rows.keys() or len(ng_rows) != count:
            raise ValueError("Different checkpoint sets")
        for key in gc_rows:
            compare_reports(gc_rows[key], ng_rows[key])
        all_rows[family] = ng_rows
        ng_groups = {g["condition"]: g for g in ng_summary["groups"]}
        groups = []
        text += [
            "",
            f"## {family} — {count} lượt chạy",
            "",
            "| Cấu hình | R@50 GC | ng-R@50 | mR@50 GC | ng-mR@50 | ng-R@100 | ng-mR@100 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for group in gc_summary["groups"]:
            name = group["condition"]
            ng_group = ng_groups[name]
            groups.append(
                {
                    "condition": name,
                    "seeds": group["seeds"],
                    "graph_constraint": group["metrics"],
                    "no_graph_constraint": ng_group["metrics"],
                }
            )
            cells = [
                group["metrics"]["R@50"],
                ng_group["metrics"]["ng-R@50"],
                group["metrics"]["mR@50"],
                ng_group["metrics"]["ng-mR@50"],
                ng_group["metrics"]["ng-R@100"],
                ng_group["metrics"]["ng-mR@100"],
            ]
            text.append(
                "| "
                + name
                + " | "
                + " | ".join(f"{c['mean']*100:.2f} ± {c['sample_std']*100:.2f}" for c in cells)
                + " |"
            )
        result["matrices"][family] = {
            "sources": [{"path": str(path), "sha256": sha256(path)} for path in (gc_path, ng_path)],
            "groups": groups,
        }
    text += [
        "",
        "## Chênh lệch giữa các mô hình trong NG",
        "",
        "Đơn vị điểm phần trăm. Paired delta theo seed 42 / 123 / 3407; đây là thống kê mô tả, không phải kiểm định ý nghĩa.",
        "",
        "| So sánh | Δ ng-R@50 trung bình | Δ ng-mR@50 trung bình | Δ ng-mR@50 từng seed |",
        "|---|---:|---:|---|",
    ]
    comparisons = [
        ("main", "gat", "mlp_semantic"),
        *[
            ("main", name, "gat")
            for name in (
                "wordnet_kg_on",
                "conceptnet_transe_kg_on",
                "conceptnet_rotate_kg_on",
                "conceptnet_complex_kg_on",
            )
        ],
        ("pair", "adjusted_off", "ce_off"),
        ("pair", "adjusted_on", "adjusted_off"),
        ("pair", "ce_on", "ce_off"),
        ("fusion", "pair_real", "baseline"),
        ("fusion", "pair_real", "pair_shuffled"),
        ("fusion", "visual_real", "baseline"),
        ("fusion", "visual_real", "visual_shuffled"),
    ]
    for family, treatment, control in comparisons:
        deltas = {}
        for metric in ("ng-R@50", "ng-mR@50", "ng-R@100", "ng-mR@100"):
            values = [
                100
                * (
                    all_rows[family][f"{treatment}-seed{s}-val5000.json"][metric]
                    - all_rows[family][f"{control}-seed{s}-val5000.json"][metric]
                )
                for s in (42, 123, 3407)
            ]
            deltas[metric] = {
                "per_seed": values,
                "mean": statistics.mean(values),
                "sample_std": statistics.stdev(values),
            }
        result["paired_effects"].append(
            {"matrix": family, "treatment": treatment, "control": control, "deltas": deltas}
        )
        text.append(
            f"| {family}: {treatment} − {control} | {deltas['ng-R@50']['mean']:+.2f} | "
            f"{deltas['ng-mR@50']['mean']:+.2f} | "
            + " / ".join(f"{v:+.2f}" for v in deltas["ng-mR@50"]["per_seed"])
            + " |"
        )
    text += [
        "",
        "Sáu run relation-prior L1/L2 vẫn bị loại. Đây là validation của checkpoint đã chọn theo legacy, "
        "không phải test độc lập. Không so ng-mR của mô hình này với mR có graph constraint của mô hình khác.",
        "",
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in (("comparison.json", json.dumps(result, indent=2) + "\n"), ("comparison.md", "\n".join(text))):
        path = args.output_dir / name
        if path.exists():
            if path.read_text() != content:
                raise FileExistsError(path)
        else:
            with path.open("x") as stream:
                stream.write(content)
    print("PASS: 54 matched checkpoints, 108 reports; comparison.json and comparison.md")


if __name__ == "__main__":
    main()
