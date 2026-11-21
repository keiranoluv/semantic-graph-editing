"""Fit-frequency breakdowns, predetermined timeline examples and paired test reporting."""

from collections import Counter
from statistics import mean, stdev

from ..contracts import relation_key
from ..intervals import temporal_iou
from ..m3.common import write_json


def timeline(cases, output):
    from PIL import Image, ImageDraw

    if not cases:
        return
    chart = Image.new("RGB", (1000, 90 * len(cases)), "white")
    draw = ImageDraw.Draw(chart)
    for index, case in enumerate(cases):
        top = 90 * index
        draw.text(
            (10, top),
            f"{case['outcome']} | {case['video']} | (s,p,o)={case['key']} | tIoU={case['tiou']:.2f}",
            fill="black",
        )
        for offset, key, color in [(24, "gt", "#25744f"), (45, "pred", "#376bb0")]:
            y = top + offset
            draw.text((10, y), key.upper(), fill="black")
            draw.line((65, y + 6, 965, y + 6), fill="#bbbbbb")
            for first, last in case[key]:
                x = 65 + 900 * first / case["frames"]
                end = 65 + 900 * (last + 1) / case["frames"]
                draw.rectangle((x, y, max(x + 1, end), y + 12), fill=color)
        draw.text((65, top + 64), "frame 0", fill="black")
        draw.text((920, top + 64), str(case["frames"] - 1), fill="black")
    chart.save(output / "case_timelines.png")


def diagnostics(store, ids, predictions, metrics, row, output):
    counts = Counter(r["predicate_id"] for v in store.ids("fit") for r in store.targets[v]["relations"])
    ordered = sorted(range(57), key=lambda i: (-counts[i], i))
    recalls = metrics["metrics"]["0.5"]["50"]["recall_per_class"]
    bins = {}
    for name, start in (("head", 0), ("mid", 19), ("tail", 38)):
        classes = ordered[start : start + 19]
        values = [recalls[i] for i in classes if recalls[i] is not None]
        bins[name] = dict(predicate_ids=classes, mR50=mean(values) if values else None)
    if (row["group"], row["model"], row["seed"]) != ("baseline", "static_gat", 17):
        return dict(fit_frequency_bins=bins)
    cases = {"hit": [], "miss": []}
    for vid, doc in zip(ids, predictions, strict=True):
        ranked = {relation_key(r): r for r in doc["predictions"][:50]}
        frames = store.videos[vid]["num_frames"]
        for gt in sorted(store.targets[vid]["relations"], key=relation_key):
            key = relation_key(gt)
            pred = ranked.get(key, {}).get("intervals", [])
            iou = temporal_iou(gt["intervals"], pred, frames) if pred else 0.0
            label = "hit" if iou >= 0.5 else "miss"
            if len(cases[label]) < 5:
                cases[label].append(
                    dict(
                        outcome=label, video=vid, key=list(key), frames=frames, gt=gt["intervals"], pred=pred, tiou=iou
                    )
                )
    examples = cases["hit"] + cases["miss"]
    timeline(examples, output)
    return dict(fit_frequency_bins=bins, case_studies=examples)


def summarize(results, out):
    baseline = {}
    lookup = {}
    followup = []
    for result in results:
        row = result["run"]
        key = row["model"], row["seed"]
        scores = result["metrics"]["metrics"]["0.5"]["50"]
        point = {k: scores[k] for k in ("mR", "R")}
        if row["group"] == "baseline":
            baseline.setdefault(row["model"], []).append(point)
            lookup[key] = point
        else:
            followup.append(dict(model=key[0], seed=key[1], original=lookup[key], min50=point))
    agg = {
        m: {k: dict(mean=mean(x[k] for x in scores), std=stdev(x[k] for x in scores)) for k in ("mR", "R")}
        for m, scores in baseline.items()
    }
    write_json(
        out / "summary.json",
        dict(heldout_videos=62, primary_metric="percent @50 tIoU 0.5", baseline=agg, post_hoc_min50_pairs=followup),
    )
    lines = [
        "# M5 · Dự đoán quan hệ trên 62 video test PVSG",
        "",
        "GT tracks và CLIP đóng băng. Điểm theo phần trăm, K=50 và tIoU=0,5. Checkpoint được chọn bằng validation. "
        "Thực nghiệm đo dự đoán quan hệ và thời gian trong phần mở rộng GML; phạm vi đầu vào giả định đã biết mask, lớp và ID.",
        "",
        "| Model | mR (mean ± SD) | R (mean ± SD) |",
        "|---|---:|---:|",
    ]
    for model, score in agg.items():
        vals = [f"{score[k]['mean']:.2f} ± {score[k]['std']:.2f}" for k in ("mR", "R")]
        lines.append(f"| {model} | {vals[0]} | {vals[1]} |")
    lines.extend(
        [
            "",
            "## Kiểm tra bổ sung về hội tụ muộn",
            "",
            "Bảy lượt được chọn sau khi xem validation vì từng dừng ở epoch 16. Bảng chính dùng protocol ban đầu. "
            "Bảng dưới so từng seed khi hoãn early stopping đến ít nhất epoch 50.",
            "",
            "| Model | Seed | mR gốc → min50 | R gốc → min50 |",
            "|---|---:|---:|---:|",
        ]
    )
    for r in followup:
        vals = [f"{r['original'][k]:.2f} → {r['min50'][k]:.2f}" for k in ("mR", "R")]
        lines.append(f"| {r['model']} | {r['seed']} | {vals[0]} | {vals[1]} |")
    lines += [
        "",
        "Ví dụ timeline từ Static GAT seed 17, chọn theo thứ tự video và tuple trong từng nhóm đúng/sai.",
        "",
        "![Timeline GT và dự đoán](case_timelines.png)",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")
