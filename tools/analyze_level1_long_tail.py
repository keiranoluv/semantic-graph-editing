"""Train-defined head/body/tail analysis of 54 fixed PredCls checkpoints, without inference."""

import argparse
import csv
import io
import json
import math
import statistics
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

from src.evaluation.reference_predcls import NG_PROTOCOL, PROTOCOL
from src.knowledge.transe.core import sha256
from tools.compare_reference_protocols import checked_reports, compare_reports

SEEDS = (42, 123, 3407)
SIZES = {"head": 10, "body": 20, "tail": 20}
DATA_FILES = ("VG-SGG-with-attri.h5", "VG-SGG-dicts-with-attri.json", "image_data.json", "splits.json")
COMPARISONS = (
    ("main", "gat", "mlp_semantic"),
    ("main", "wordnet_kg_on", "kg_off"),
    ("main", "wordnet_kg_on", "wordnet_edges_off"),
    ("main", "conceptnet_transe_kg_on", "conceptnet_transe_edges_off"),
    *[
        ("main", name, "gat")
        for name in ("wordnet_kg_on", "conceptnet_transe_kg_on", "conceptnet_rotate_kg_on", "conceptnet_complex_kg_on")
    ],
    ("pair", "adjusted_off", "ce_off"),
    ("pair", "adjusted_on", "adjusted_off"),
    ("pair", "ce_on", "ce_off"),
    ("fusion", "pair_real", "baseline"),
    ("fusion", "pair_real", "pair_shuffled"),
    ("fusion", "visual_real", "baseline"),
    ("fusion", "visual_real", "visual_shuffled"),
)


def train_predicate_counts(h5, train_rows):
    """Count raw annotation rows, including duplicates/multi-label pairs, only in train."""
    if not train_rows or len(set(train_rows)) != len(train_rows):
        raise ValueError("Expected nonempty, unique train image rows")
    counts = Counter({c: 0 for c in range(1, 51)})
    for row in train_rows:
        if not 0 <= row < len(h5["img_to_first_rel"]):
            raise ValueError("Train image row outside HDF5")
        first = int(np.asarray(h5["img_to_first_rel"][row]).item())
        last = int(np.asarray(h5["img_to_last_rel"][row]).item())
        if first == -1 and last == -1:
            continue
        if first < 0 or last < first or last >= len(h5["predicates"]):
            raise ValueError("Invalid train relationship range")
        labels = h5["predicates"][first : last + 1].reshape(-1)
        if ((labels < 1) | (labels > 50)).any():
            raise ValueError("Predicate outside foreground vocabulary")
        counts.update(int(c) for c in labels)
    return dict(counts)


def define_groups(counts, names):
    if set(counts) != set(range(1, 51)) or set(names) != set(counts):
        raise ValueError("Expected all 50 predicates")
    if any(type(n) is not int or n < 0 for n in counts.values()) or sum(counts.values()) == 0:
        raise ValueError("Invalid train counts")
    ordered = sorted(counts, key=lambda c: (-counts[c], c))
    groups = {"head": ordered[:10], "body": ordered[10:30], "tail": ordered[30:]}
    total = sum(counts.values())
    return {
        "schema_version": 1,
        "frequency_source": "all_raw_train_annotation_rows_including_duplicates_and_multilabel_pairs",
        "sort_rule": "descending_train_count_then_ascending_predicate_id",
        "group_rule": "head_ranks_1_to_10_body_11_to_30_tail_31_to_50",
        "selection_note": "project_analysis_convention_fixed_before_group_metric_aggregation_not_a_universal_benchmark_partition",
        "total_train_annotations": total,
        "groups": groups,
        "group_train_support": {
            group: {
                "classes": len(ids),
                "annotations": sum(counts[c] for c in ids),
                "annotation_share": sum(counts[c] for c in ids) / total,
            }
            for group, ids in groups.items()
        },
        "predicates": [
            {
                "id": c,
                "name": names[c],
                "train_rank": rank,
                "train_annotations": counts[c],
                "group": next(g for g, ids in groups.items() if c in ids),
            }
            for rank, c in enumerate(ordered, 1)
        ],
    }


def grouped_recall(report, groups, prefix):
    """Macro-average existing image-macro class recalls; never reweight by support."""
    ids = [c for group in groups.values() for c in group]
    if sorted(ids) != list(range(1, 51)) or any(len(groups[g]) != n for g, n in SIZES.items()):
        raise ValueError("Groups must partition 50 classes as 10/20/20")
    per = report["per_predicate"]
    if set(per) != {str(c) for c in ids} or report["num_predicates_for_mean"] != 50:
        raise ValueError("Expected fixed 50-class reference report")
    for row in per.values():
        if row["support"] < row["images_with_gt"] or row["images_with_gt"] < 0:
            raise ValueError("Invalid class support")
        for k in (50, 100):
            score = row[f"{prefix}R@{k}"]
            if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Invalid class recall")
            if row["images_with_gt"] == 0 and (score != 0 or row["support"] != 0):
                raise ValueError("Absent classes must contribute zero")
        if row[f"{prefix}R@100"] + 1e-12 < row[f"{prefix}R@50"]:
            raise ValueError("Recall must be monotone in K")
    result = {
        group: {f"{prefix}mR@{k}": statistics.mean(per[str(c)][f"{prefix}R@{k}"] for c in members) for k in (50, 100)}
        for group, members in groups.items()
    }
    for k in (50, 100):
        recovered = sum(len(groups[g]) * scores[f"{prefix}mR@{k}"] for g, scores in result.items()) / 50
        if not math.isclose(recovered, report[f"{prefix}mR@{k}"], rel_tol=0, abs_tol=1e-12):
            raise ValueError("Group recalls do not reconstruct overall mR")
    return result


def stats(values):
    return {"per_seed": values, "mean": statistics.mean(values), "sample_std": statistics.stdev(values)}


def write_fixed(path, content):
    if path.exists():
        if path.read_text() != content:
            raise FileExistsError(f"Refusing changed output; choose a new --output-dir: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            stream.write(content)


def encode(value):
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/vg150"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/level1/long-tail-reference-v1"))
    parser.add_argument(
        "--groups-only", action="store_true", help="Freeze train-defined groups before reading evaluation reports"
    )
    args = parser.parse_args()
    splits = json.loads((args.root / "splits.json").read_text())
    if len(splits["train"]) != 57723 or len(splits["val"]) != 5000:
        raise ValueError("Expected the frozen full VG150 train/validation split")
    if set(splits["train"]) & (set(splits["val"]) | set(splits["test"])):
        raise ValueError("Train split overlaps held-out rows")
    vocab = json.loads((args.root / "VG-SGG-dicts-with-attri.json").read_text())
    names = {v: k for k, v in vocab["predicate_to_idx"].items()}
    with h5py.File(args.root / "VG-SGG-with-attri.h5", "r") as h5:
        counts = train_predicate_counts(h5, splits["train"])
    definition = define_groups(counts, names)
    definition["train_images"] = len(splits["train"])
    definition["data_sha256"] = {name: sha256(args.root / name) for name in DATA_FILES}
    definition_path = args.output_dir / "predicate-groups.json"
    # Persist the grouping before any model scores are read.
    write_fixed(definition_path, encode(definition))
    if args.groups_only:
        print(encode(definition["group_train_support"]))
        return
    result = {
        "status": "PASS",
        "split": "val",
        "images_evaluated": 5000,
        "seeds": list(SEEDS),
        "checkpoints": 54,
        "source_reports": 108,
        "condition_count": 18,
        "protocols": {"GC": PROTOCOL, "NG": NG_PROTOCOL},
        "group_definition": {"path": str(definition_path), "sha256": sha256(definition_path)},
        "metric_definition": "macro_mean_of_saved_image_macro_predicate_recalls_within_fixed_group",
        "ranking_scope": "original_topK_over_all_predictions_in_each_image_not_reranked_within_group",
        "absent_class_policy": "retain_class_in_fixed_denominator_and_assign_zero",
        "metrics_unit": "fraction_0_to_1",
        "paired_delta_unit": "percentage_points",
        "std_definition": "sample_standard_deviation_ddof_1",
        "training_or_inference_performed": False,
        "interpretation": "posthoc_validation_analysis_of_fixed_legacy_selected_checkpoints_not_reselected",
        "excluded_conditions": ["fusion/relation_real", "fusion/relation_shuffled"],
        "analysis_source_sha256": {
            str(p): sha256(p)
            for p in (Path(__file__).relative_to(Path.cwd()), Path("tools/compare_reference_protocols.py"))
        },
        "summary_sources": [],
        "runs": [],
        "conditions": [],
        "paired_effects": [],
    }
    run_lookup = {}
    per_class_rows = []
    support_reference = None
    for family, number in (("main", 27), ("pair", 12), ("fusion", 15)):
        suffix = "" if family == "main" else "-" + family
        reports = {}
        for mode, folder, protocol in (("GC", "reference-v1", PROTOCOL), ("NG", "reference-ng-v1", NG_PROTOCOL)):
            summary_path = Path(f"reports/level1/{folder}{suffix}/{number}run-val5000-summary.json")
            summary, rows = checked_reports(summary_path, protocol)
            result["summary_sources"].append({"path": str(summary_path), "sha256": sha256(summary_path)})
            reports[mode] = {"summary": summary, "rows": rows, "folder": summary_path.parent}
        if reports["GC"]["rows"].keys() != reports["NG"]["rows"].keys() or len(reports["GC"]["rows"]) != number:
            raise ValueError("Mismatched source checkpoint sets")
        for group in reports["GC"]["summary"]["groups"]:
            condition = group["condition"]
            for seed in SEEDS:
                filename = f"{condition}-seed{seed}-val5000.json"
                gc, ng = (reports[m]["rows"][filename] for m in ("GC", "NG"))
                compare_reports(gc, ng)
                if gc["data_sha256"] != definition["data_sha256"]:
                    raise ValueError("Training frequency inputs differ from model evaluation inputs")
                support = {c: (r["support"], r["images_with_gt"]) for c, r in gc["per_predicate"].items()}
                if support_reference is None:
                    support_reference = support
                if support != support_reference:
                    raise ValueError("Validation support differs across runs")
                item = {
                    "matrix": family,
                    "condition": condition,
                    "seed": seed,
                    "checkpoint_sha256": gc["checkpoint_sha256"],
                    "metrics": {},
                    "sources": {},
                }
                for mode, row, prefix in (("GC", gc, ""), ("NG", ng, "ng-")):
                    item["metrics"][mode] = grouped_recall(row, definition["groups"], prefix)
                    source = reports[mode]["folder"] / filename
                    item["sources"][mode] = {"path": str(source), "sha256": sha256(source)}
                run_lookup[(family, condition, seed)] = item
                result["runs"].append(item)
            aggregate = {"matrix": family, "condition": condition, "metrics": {}}
            for mode, prefix in (("GC", ""), ("NG", "ng-")):
                aggregate["metrics"][mode] = {
                    g: {
                        f"{prefix}mR@{k}": stats(
                            [run_lookup[(family, condition, s)]["metrics"][mode][g][f"{prefix}mR@{k}"] for s in SEEDS]
                        )
                        for k in (50, 100)
                    }
                    for g in SIZES
                }
            result["conditions"].append(aggregate)
            for predicate in definition["predicates"]:
                c = str(predicate["id"])
                record = {
                    "matrix": family,
                    "condition": condition,
                    **predicate,
                    "validation_annotations": support_reference[c][0],
                    "validation_images": support_reference[c][1],
                }
                for mode, prefix in (("GC", ""), ("NG", "ng-")):
                    for k in (50, 100):
                        values = [
                            reports[mode]["rows"][f"{condition}-seed{s}-val5000.json"]["per_predicate"][c][
                                f"{prefix}R@{k}"
                            ]
                            for s in SEEDS
                        ]
                        for statistic in ("mean", "sample_std"):
                            record[f"{mode}_R@{k}_{statistic}"] = stats(values)[statistic]
                per_class_rows.append(record)
    if len(result["runs"]) != 54 or len({r["checkpoint_sha256"] for r in result["runs"]}) != 54:
        raise ValueError("Expected exactly 54 unique checkpoints")
    for family, treatment, control in COMPARISONS:
        effect = {"matrix": family, "treatment": treatment, "control": control, "deltas": {}}
        for mode, prefix in (("GC", ""), ("NG", "ng-")):
            effect["deltas"][mode] = {
                g: {
                    f"{prefix}mR@{k}": stats(
                        [
                            100
                            * (
                                run_lookup[(family, treatment, s)]["metrics"][mode][g][f"{prefix}mR@{k}"]
                                - run_lookup[(family, control, s)]["metrics"][mode][g][f"{prefix}mR@{k}"]
                            )
                            for s in SEEDS
                        ]
                    )
                    for k in (50, 100)
                }
                for g in SIZES
            }
        result["paired_effects"].append(effect)
    result["validation_support"] = {
        g: {
            "annotations": sum(support_reference[str(c)][0] for c in ids),
            "classes_with_gt": sum(support_reference[str(c)][0] > 0 for c in ids),
            "absent_classes": [names[c] for c in ids if not support_reference[str(c)][0]],
        }
        for g, ids in definition["groups"].items()
    }
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=list(per_class_rows[0]))
    writer.writeheader()
    writer.writerows(per_class_rows)
    write_fixed(args.output_dir / "per-predicate.csv", csv_buffer.getvalue().replace("\r\n", "\n"))
    result["per_predicate_csv_sha256"] = sha256(args.output_dir / "per-predicate.csv")
    write_fixed(args.output_dir / "analysis.json", encode(result))
    write_fixed(args.output_dir / "analysis.md", render_markdown(result, definition))
    print("PASS: train-defined 10/20/20 groups; 54 checkpoints / 108 reports; 216 overall-mR reconstruction checks")
    print(encode(result["validation_support"]))


def render_markdown(result, definition):
    lines = [
        "# Long-tail Mức 1 — cùng checkpoint, cùng ranking, chia nhóm theo train",
        "",
        "54 checkpoint / 18 cấu hình × 3 seed (42, 123, 3407), cùng 5.000 ảnh validation. "
        "GC và NG được tổng hợp riêng từ report đã có; không train, không inference lại, không chọn lại checkpoint.",
        "",
        "## Cách chia và cách tính",
        "",
        "Đếm tất cả annotation quan hệ trong 57.723 ảnh train, giữ dòng trùng và cặp nhiều predicate. "
        "Xếp tần suất giảm dần; đồng hạng xếp predicate ID tăng dần. Head = hạng 1–10, body = 11–30, tail = 31–50. "
        "Đây là quy ước phân tích của dự án, không khẳng định mọi paper dùng cùng nhóm.",
        "",
        "mR nhóm = trung bình recall theo ảnh của từng predicate trong nhóm. Lớp vắng GT validation vẫn đóng góp 0 "
        "và nằm trong mẫu số cố định. Top-K giữ nguyên danh sách dự đoán của **toàn ảnh**; không lọc nhóm rồi lấy lại top-K. "
        "Không tính được R nhóm theo ảnh chỉ từ các report tổng hợp này, nên chỉ báo **mR theo nhóm**.",
        "",
        "Với từng checkpoint/protocol/K, đã kiểm `0,2 × head-mR + 0,4 × body-mR + 0,4 × tail-mR = mR toàn bộ 50 lớp` "
        "(sai số ≤ 1e-12). Không lấy trung bình không trọng số của ba nhóm có kích thước khác nhau.",
        "",
        "| Nhóm | Số lớp | Annotation train | Tỷ trọng train | Annotation val | Lớp có GT val |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for g, support in definition["group_train_support"].items():
        val = result["validation_support"][g]
        lines.append(
            f"| {g} | {support['classes']} | {support['annotations']} | {support['annotation_share']:.2%} | {val['annotations']} | {val['classes_with_gt']} |"
        )
        if val["absent_classes"]:
            lines.append("")
            lines.append("Lớp vắng GT validation, vẫn tính 0 trong nhóm: " + ", ".join(val["absent_classes"]) + ".")
    lines += ["", "## Kết quả — đơn vị %, mean ± sample std", ""]
    for family in ("main", "pair", "fusion"):
        for mode, prefix in (("GC", ""), ("NG", "ng-")):
            lines += [
                f"### {family} / {mode}",
                "",
                f"| Cấu hình | Head {prefix}mR@50 | Body {prefix}mR@50 | Tail {prefix}mR@50 | Head {prefix}mR@100 | Body {prefix}mR@100 | Tail {prefix}mR@100 |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
            for condition in result["conditions"]:
                if condition["matrix"] != family:
                    continue
                values = [condition["metrics"][mode][g][f"{prefix}mR@{k}"] for k in (50, 100) for g in SIZES]
                lines.append(
                    "| "
                    + condition["condition"]
                    + " | "
                    + " | ".join(f"{v['mean']*100:.2f} ± {v['sample_std']*100:.2f}" for v in values)
                    + " |"
                )
            lines.append("")
    lines += [
        "## Chênh lệch ghép cặp theo seed",
        "",
        "Đơn vị điểm phần trăm. Dãy seed theo thứ tự 42 / 123 / 3407. Dấu dương nghĩa là treatment cao hơn control. "
        "Đây là thống kê mô tả trên validation, không phải kiểm định ý nghĩa thống kê.",
        "",
    ]
    for mode, prefix in (("GC", ""), ("NG", "ng-")):
        lines += [
            f"### {mode}",
            "",
            "| Treatment − control | Δ head@50 | Δ body@50 | Δ tail@50 | Δ tail@50 từng seed | Δ tail@100 |",
            "|---|---:|---:|---:|---|---:|",
        ]
        for effect in result["paired_effects"]:
            d = effect["deltas"][mode]
            values = [d[g][f"{prefix}mR@50"]["mean"] for g in SIZES]
            seeds = d["tail"][f"{prefix}mR@50"]["per_seed"]
            lines.append(
                f"| {effect['matrix']}: {effect['treatment']} − {effect['control']} | "
                + " | ".join(f"{v:+.2f}" for v in values)
                + " | "
                + " / ".join(f"{v:+.2f}" for v in seeds)
                + f" | {d['tail'][f'{prefix}mR@100']['mean']:+.2f} |"
            )
        lines.append("")
    lines += [
        "## Danh sách đủ 50 predicate và tần suất train",
        "",
        "| Hạng train | ID | Predicate | Nhóm | Annotation train |",
        "|---:|---:|---|---|---:|",
    ]
    for p in definition["predicates"]:
        lines.append(f"| {p['train_rank']} | {p['id']} | {p['name']} | {p['group']} | {p['train_annotations']} |")
    lines += [
        "",
        "## Phạm vi và tái lập",
        "",
        "Sáu run relation-prior L1/L2 tiếp tục bị loại. Nhóm head/body/tail được khóa từ train trước khi tổng hợp metric; "
        "đây vẫn là phân tích hậu kiểm của thí nghiệm cũ, không phải giả thuyết đã đăng ký trước lúc train. "
        "Các lớp hiếm có ít annotation/ảnh validation nên cần đọc số support trong CSV cùng với mean/std.",
        "",
        "[Nhóm đã khóa](predicate-groups.json) · [Số liệu, từng seed, paired delta và hash nguồn](analysis.json) · "
        "[Recall từng predicate, đơn vị fraction 0–1](per-predicate.csv).",
        "",
        "```bash",
        "myenv/bin/python -m tools.analyze_level1_long_tail --groups-only",
        "myenv/bin/python -m tools.analyze_level1_long_tail",
        "```",
        "",
        "Công cụ kiểm hash report/checkpoint, data hash, support giữa các run, mean/std gốc và tái dựng mR. "
        "Chạy lại giữ nguyên output nếu giống hoàn toàn; từ chối ghi đè nếu khác. Dùng `--output-dir` mới cho phân tích phiên bản khác.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
