"""Human outcomes and historical baseline scores enter only at this stage."""

import csv
import tempfile
from pathlib import Path

import numpy as np

from tools.level3_analysis_v1.core import cluster_indices, rho

from .protocol import RELEASE, atomic, by_id, checked, digest, read, sha256

BASE = {"graph_consistency": 0.1767, "clipscore": 0.0561, "negative_lpips": 0.1962}
COMPONENTS = {"fulfillment": "edit_fulfillment", "preservation": "human_preservation", "J": "overall_pass"}


def correlation(rows, x, y):
    return rho([r[x] for r in rows], [int(r[y]) for r in rows])


def references(cfg, cases):
    baselines = by_id(read(checked(cfg["baselines"])))
    pack = read(checked(cfg["labels"]))
    labels = by_id(pack["reviews"])
    if (
        pack["cohort_sha256"] != cfg["cohort"]["sha256"]
        or labels.keys() != baselines.keys()
        or labels.keys() != by_id(cases).keys()
    ):
        raise ValueError("Unmatched cohort IDs or labels")
    rows = []
    for case in cases:
        tid = case["task_id"]
        baseline, label = baselines[tid], labels[tid]
        if (
            case["image_id"] != baseline["image_id"]
            or case["binding"] != label["binding"]
            or label["status"] != "reviewed"
            or any(
                type(label[k]) is not bool or baseline[k] is not label[k] for k in ("edit_fulfillment", "preservation")
            )
            or baseline["overall_pass"] is not (label["edit_fulfillment"] and label["preservation"])
            or baseline["clipscore"] is None
            or baseline["negative_lpips"] != -baseline["lpips"]
        ):
            raise ValueError(f"Baseline / human label mismatch: {tid}")
        rows.append(
            {
                "task_id": tid,
                "image_id": case["image_id"],
                "edit_fulfillment": label["edit_fulfillment"],
                "human_preservation": label["preservation"],
                "overall_pass": baseline["overall_pass"],
                **{k: baseline[k] for k in BASE},
            }
        )
    common = [r for r in rows if r["graph_consistency"] is not None]
    if (len(common), len({r["image_id"] for r in common}), sum(r["overall_pass"] for r in common)) != (91, 62, 82):
        raise ValueError("Historical 91-pair panel mismatch")
    observed = {k: round(correlation(common, k, "overall_pass"), 4) for k in BASE}
    if observed != BASE:
        raise ValueError(f"Baseline regression: {observed}")
    return rows


def estimates(rows, metrics, compare):
    values = {k: correlation(rows, k, label) for k, label in metrics.items()}
    if compare:
        for k in BASE:
            values["J_minus_" + k] = None if values["J"] is None or values[k] is None else values["J"] - values[k]
    return values


def panel(rows, cfg, compare=False):
    selected = [r for r in rows if r["status"] == "ok"]
    metrics = dict(COMPONENTS)
    if compare:
        metrics.update(dict.fromkeys(BASE, "overall_pass"))
    observed = estimates(selected, metrics, compare)
    samples = {k: [] for k in observed}
    rng = np.random.default_rng(cfg["seed"])
    n = cfg["bootstrap_replicates"]
    for _ in range(n):
        take = [selected[i] for i in cluster_indices(selected, rng)]
        for k, value in estimates(take, metrics, compare).items():
            if value is not None:
                samples[k].append(value)
    return {
        "n": len(selected),
        "total": len(rows),
        "invalid_format": len(rows) - len(selected),
        "sources": len({r["image_id"] for r in rows}),
        "valid_sources": len({r["image_id"] for r in selected}),
        "human_overall_pass": sum(r["overall_pass"] for r in rows),
        "valid_ids": [r["task_id"] for r in selected],
        "statistics": {
            k: {
                "estimate": observed[k],
                "ci95": np.quantile(v, [0.025, 0.975]).tolist() if v else None,
                "valid_replicates": len(v),
                "undefined_replicates": n - len(v),
            }
            for k, v in samples.items()
        },
    }


def markdown(result):
    def fmt(x):
        return "không xác định" if x is None else f"{x:.4f}"

    lines = [
        "# Đối chứng Qwen3-VL trên VN100",
        "",
        "Spearman với nhãn tay. Điểm J = fulfillment × preservation / 16. Bootstrap lấy mẫu theo ảnh nguồn, "
        f"{result['bootstrap']['replicates']:,} lượt ghép cặp, seed {result['bootstrap']['seed']}. Δρ là chênh lệch giữa hai tương quan với cùng nhãn tổng hợp.",
        "",
    ]
    names = {"common91": "91 cặp có đủ baseline", "all200": "Toàn bộ 200 cặp", "abstain109": "109 cặp G khuyết"}
    display = {
        "fulfillment": "Đúng yêu cầu",
        "preservation": "Bảo toàn",
        "J": "J",
        "graph_consistency": "G",
        "clipscore": "CLIPScore",
        "negative_lpips": "−LPIPS",
    }
    for group, data in result["panels"].items():
        lines.extend(
            [
                f"## {names[group]}",
                "",
                f"Qwen chấm được **{data['n']}/{data['total']}** cặp, lỗi định dạng {data['invalid_format']}. "
                f"Tập có {data['sources']} ảnh nguồn và {data['human_overall_pass']} cặp đạt cả hai tiêu chí theo nhãn tay.",
                "",
                "| Chỉ số | ρ hoặc Δρ | Khoảng 95% | Lượt bootstrap hợp lệ |",
                "|---|---:|---|---:|",
            ]
        )
        for k, stat in data["statistics"].items():
            name = display[k] if k in display else "J trừ " + display[k.removeprefix("J_minus_")]
            ci = stat["ci95"]
            interval = "không xác định" if ci is None else f"[{fmt(ci[0])}, {fmt(ci[1])}]"
            lines.append(f"| {name} | {fmt(stat['estimate'])} | {interval} | {stat['valid_replicates']} |")
        lines.append("")
    lines.extend(
        [
            "Nhãn do một người duyệt. Có 25/200 cặp trượt ít nhất một tiêu chí. "
            "Nhãn cuối giữ quy ước trường trống được người duyệt xác nhận là đạt. "
            "Đối chứng greedy dùng một seed trên chính bộ ảnh đã phân tích. Khoảng bootstrap phản ánh bất định "
            "trong từng tập. Tập 91 được chọn theo độ phủ của đồ thị nên kết quả ở đó chỉ áp dụng cho tập này.",
            "",
        ]
    )
    return "\n".join(lines)


def build(folder, predictions, refs, cfg, header):
    lookup = by_id(predictions)
    if lookup.keys() != by_id(refs).keys():
        raise ValueError("Need one attempted prediction per VN100 ID")
    rows = [{**ref, **lookup[ref["task_id"]]["scores"]} for ref in refs]
    result = {
        "protocol": digest(header),
        "release_sha256": sha256(RELEASE),
        "score_receipt_sha256": sha256(folder / "complete.json"),
        "labels": cfg["labels"],
        "baselines": cfg["baselines"],
        "original_91_spearman": BASE,
        "bootstrap": {
            "cluster": "image_id",
            "paired": True,
            "seed": cfg["seed"],
            "replicates": cfg["bootstrap_replicates"],
            "interval": "percentile_95",
        },
        "panels": {
            "common91": panel([r for r in rows if r["graph_consistency"] is not None], cfg, True),
            "all200": panel(rows, cfg),
            "abstain109": panel([r for r in rows if r["graph_consistency"] is None], cfg),
        },
    }
    out = folder / "report"
    with tempfile.TemporaryDirectory(dir=folder) as tmp:
        stage = Path(tmp) / "report"
        atomic(stage / "summary.json", result)
        with (stage / "results.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (stage / "report.md").write_text(markdown(result), encoding="utf-8")
        if out.exists():
            if any(sha256(p) != sha256(out / p.name) for p in stage.iterdir()):
                raise ValueError("Existing report differs")
        else:
            stage.rename(out)
    return out
