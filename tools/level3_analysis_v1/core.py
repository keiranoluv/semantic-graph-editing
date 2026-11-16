"""Read immutable evaluation artifacts, bootstrap by source, and render evidence."""
import hashlib
import html
import importlib.metadata
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from tools.level3_evaluation.report import METRICS, comparison, disagreements

ROOT = Path(__file__).resolve().parents[2]
NAMES = {"graph_consistency": "Graph G", "clipscore": "CLIPScore", "negative_lpips": "−LPIPS"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def rho(x, y):
    if len(x) < 2 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def cluster_indices(rows, rng):
    """Draw whole source-image clusters with replacement, keeping every paired edit."""
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(row["image_id"], []).append(i)
    keys = sorted(groups)
    if not keys:
        return []
    return [i for g in rng.integers(len(keys), size=len(keys)) for i in groups[keys[g]]]


def bootstrap(rows, replicates=2000, seed=42):
    if replicates < 1:
        raise ValueError("positive_replicates_required")
    common = [r for r in rows if all(r.get(k) is not None for k in (*METRICS, "overall_pass"))]
    rng = np.random.default_rng(seed)
    samples = {k: [] for k in (*METRICS, "G_minus_CLIPScore", "G_minus_negative_LPIPS")}
    for _ in range(replicates):
        selected = [common[i] for i in cluster_indices(common, rng)]
        y = [int(r["overall_pass"]) for r in selected]
        values = {k: rho([r[k] for r in selected], y) for k in METRICS}
        for k, value in values.items():
            if value is not None:
                samples[k].append(value)
        for name, baseline in (("G_minus_CLIPScore", "clipscore"), ("G_minus_negative_LPIPS", "negative_lpips")):
            if values["graph_consistency"] is not None and values[baseline] is not None:
                samples[name].append(values["graph_consistency"] - values[baseline])
    return {
        "method": "paired source-image cluster bootstrap; percentile 95% intervals; no independent-pair p-values",
        "scope": "exploratory uncertainty conditional on the common evaluable development cohort",
        "seed": seed,
        "replicates": replicates,
        "common_pairs": len(common),
        "source_clusters": len({r["image_id"] for r in common}),
        "intervals": {
            k: {
                "valid_replicates": len(v),
                "undefined_replicates": replicates - len(v),
                "ci95": [float(x) for x in np.quantile(v, [0.025, 0.975])] if v else None,
            }
            for k, v in samples.items()
        },
    }


def validate_notes(notes):
    if notes["schema"] != "level3-visual-review-v1" or notes["human_ground_truth"] is not False:
        raise ValueError("visual_review_must_be_assistant_analysis")
    for path, expected in notes["inputs_sha256"].items():
        if sha(ROOT / path) != expected:
            raise ValueError("analysis_input_changed:" + path)
    report = ROOT / notes["report"]
    rows = read(report / "results.json")
    summary = read(report / "summary.json")
    if summary["preliminary"] or comparison(rows) != summary["all"]:
        raise ValueError("inconsistent_or_partial_report")
    selected = disagreements(rows)
    if [x["task_id"] for x in notes["cases"]] != [x["task_id"] for x in selected]:
        raise ValueError("visual_reviews_must_match_frozen_top10_selection")
    for note in notes["cases"]:
        for field in ("visual_observation", "graph_evidence", "interpretation", "limitation"):
            if not isinstance(note.get(field), str) or not note[field].strip():
                raise ValueError("missing_visual_review:" + field)
    return report, rows, summary, selected


def copy_media(source, dest, expected):
    source = Path(source)
    if sha(source) != expected:
        raise ValueError("image_or_graph_changed:" + str(source))
    name = "media/" + expected + source.suffix
    target = dest / name
    target.parent.mkdir(exist_ok=True)
    if not target.exists():
        try:
            os.link(source, target)
        except OSError:
            shutil.copyfile(source, target)
    return name


def clip_preview(path, dest):
    """Invert normalization of the actual frozen processor; no model forward pass."""
    from PIL import Image, ImageOps
    from transformers import CLIPProcessor

    processor = CLIPProcessor.from_pretrained(
        ROOT / "data/cache/level3-task3-v1/clip", local_files_only=True, use_fast=False
    ).image_processor
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        pixels = processor(images=im, return_tensors="np")["pixel_values"][0].transpose(1, 2, 0)
    pixels = (pixels * np.array(processor.image_std) + np.array(processor.image_mean)) * 255
    Image.fromarray(pixels.round().clip(0, 255).astype("uint8")).save(dest)
    return processor.to_dict()


def label_counts(rows):
    return {
        "n": len(rows),
        "overall_pass": sum(r["overall_pass"] for r in rows),
        "overall_fail": sum(not r["overall_pass"] for r in rows),
        "edit_fail": sum(not r["edit_fulfillment"] for r in rows),
        "preservation_fail": sum(not r["preservation"] for r in rows),
    }


def graph_svg(graph, region=None):
    w, h = graph["image"]["width"], graph["image"]["height"]
    shapes = []
    for key, node in graph["grounding"]["nodes"].items():
        x, y, x2, y2 = node["bbox_xyxy"]
        title = html.escape(f'{key}: {node["label"]}, confidence={node["confidence"]}')
        shapes.append(
            f'<g><title>{title}</title><rect x="{x}" y="{y}" width="{x2-x}" height="{y2-y}"/>'
            f'<text x="{x}" y="{max(y, h/40)}" font-size="{max(w,h)/55}">{html.escape(key+": "+node["label"])}</text></g>'
        )
    if region:
        x, y, x2, y2 = region
        shapes.append(
            f'<rect class="roi" x="{x*w}" y="{y*h}" width="{(x2-x)*w}" height="{(y2-y)*h}"><title>Vùng edit định trước</title></rect>'
        )
    return f'<svg class="boxes" viewBox="0 0 {w} {h}" aria-label="Bounding boxes REACT++">' + "".join(shapes) + "</svg>"


def render(pack):
    esc = html.escape

    def fmt(x):
        return "Không xác định" if x is None else f"{x:.3f}"

    def table(headers, rows):
        return (
            "<table><thead><tr>"
            + "".join(f"<th>{esc(str(h))}</th>" for h in headers)
            + "</tr></thead><tbody>"
            + "".join("<tr>" + "".join(f"<td>{esc(str(c))}</td>" for c in row) + "</tr>" for row in rows)
            + "</tbody></table>"
        )

    summary = pack["summary"]["all"]
    primary = []
    for metric, result in summary["common_case_correlations"].items():
        ci = pack["bootstrap"]["intervals"][metric]["ci95"]
        primary.append(
            [NAMES[metric], result["n"], fmt(result["spearman"]), " – ".join(map(fmt, ci)) if ci else "Không xác định"]
        )
    content = "<h1>Level 3 · kết quả và 10 ca bất đồng</h1><p>Qwen-Image-2.1 · REACT++ · seed 42. Phân tích đã chốt nhãn, không tinh chỉnh theo nhãn.</p>"
    content += "<p><b>200 cặp đã có nhãn: 175 đạt cả hai, 25 không đạt ít nhất một tiêu chí.</b> Graph G chấm được 91/200 (45,5%); 109 cặp không có điểm G, không quy thành 0.</p>"
    content += table(["Chỉ số", "Cùng số cặp", "Spearman với nhãn tổng hợp", "Khoảng bootstrap 95%"], primary)
    content += "<p>Khoảng bootstrap lấy lại mẫu theo 62 ảnh nguồn, 2.000 lượt, seed 42; chỉ phản ánh tập 91 cặp có điểm. Chưa có bằng chứng graph tốt hơn −LPIPS. LPIPS đo mức thay đổi thị giác, không kiểm tra trực tiếp câu lệnh.</p>"
    content += "<p>Nhãn tổng hợp = đúng yêu cầu VÀ bảo toàn. Theo xác nhận của người duyệt, 287 trường để trống ở 144 cặp được hiểu là đạt; giữ nguyên mọi nhãn điền rõ. Nhận xét ảnh bên dưới là phân tích của assistant, không thay thế nhãn tay.</p>"
    content += table(
        ["Loại edit", "Tổng", "Có G", "ρ Graph", "ρ CLIPScore", "ρ −LPIPS"],
        [
            [op, v["total_pairs"], v["common_n"], *[fmt(v["common_case_correlations"][m]["spearman"]) for m in METRICS]]
            for op, v in pack["summary"]["by_operation"].items()
        ],
    )
    content += "<p>10 yêu cầu đổi màu nằm ngoài khả năng biểu diễn: E/G không xác định. E so với đúng yêu cầu: ρ=0,201 (162 cặp); P so với bảo toàn: ρ=0,030 (109 cặp). Không so trực tiếp các hệ số từ các tập có số mẫu khác nhau.</p>"
    content += "<p>Chọn 10 chênh lệch thứ hạng lớn nhất trên 91 cặp theo quy tắc đã khóa. Cả 10 thuộc loại thêm đối tượng; đây không phải mẫu ngẫu nhiên để ước lượng độ chính xác SGG. Điểm cao/thấp dưới đây là thứ hạng tương đối, không phải ngưỡng PASS/FAIL.</p>"
    content += "<nav>" + " · ".join(f'<a href="#case-{c["index"]}">#{c["index"]}</a>' for c in pack["cases"]) + "</nav>"
    for case in pack["cases"]:
        score = case["candidate"]
        content += f'<article id="case-{case["index"]}"><h2>#{case["index"]} · {esc(case["task_id"])}</h2><p>{esc(case["instruction"])}</p>'
        content += (
            "<p><b>Nhãn người duyệt:</b> Đúng yêu cầu: "
            + ("Đạt" if score["edit_fulfillment"] else "Không đạt")
            + " · Bảo toàn: "
            + ("Đạt" if score["preservation"] else "Không đạt")
            + '</p><div class="pair">'
        )
        for role, title in (("source", "Nguồn"), ("target", "Qwen")):
            src = case["media"][role]
            content += (
                f'<figure><figcaption>{title} · <a href="{esc(src)}" target="_blank" rel="noopener">Mở ảnh gốc</a></figcaption><div class="photo"><img src="{esc(src)}" alt="{title} mẫu {case["index"]}">'
                + graph_svg(case["graphs"][role], case["grounding"]["region"] if role == "target" else None)
                + "</div></figure>"
            )
        content += "</div>" + table(
            ["E", "P", "G", "CLIPScore", "LPIPS (thấp = ít thay đổi)"],
            [[fmt(score[k]) for k in ("edit_score", "preservation_score", "graph_consistency", "clipscore", "lpips")]],
        )
        content += (
            "<p>Phân vị thứ hạng: "
            + " · ".join(f'{NAMES[k]} {score["rank_percentiles"][k]:.1%}' for k in METRICS)
            + "</p>"
        )
        for field, title in (
            ("visual_observation", "Quan sát ảnh"),
            ("graph_evidence", "Bằng chứng graph"),
            ("interpretation", "Giải thích bất đồng"),
            ("limitation", "Giới hạn nhận xét"),
        ):
            content += f"<p><b>{title}:</b> {esc(case[field])}</p>"
        if "clip_preview" in case:
            content += f'<figure class="crop"><figcaption>Đầu vào CLIP 224×224, khôi phục từ pixel sau preprocessing; không chạy lại model.</figcaption><img src="{esc(case["clip_preview"])}" alt="Đầu vào CLIP mẫu {case["index"]}"></figure>'
        evidence = {"grounding": case["grounding"], "scores": case["scoring_evidence"], "graphs": case["graphs"]}
        content += (
            "<details><summary>Xem node, cạnh có hướng, confidence và phép ghép</summary><pre>"
            + esc(json.dumps(evidence, ensure_ascii=False, indent=2))
            + "</pre></details></article>"
        )
    content += '<p><a href="analysis.json">Tải toàn bộ số liệu và provenance JSON</a> · <a href="results.csv">Bảng 200 cặp CSV</a></p>'
    template = Path(__file__).with_name("viewer.html").read_text(encoding="utf-8")
    return template.replace("<!--CONTENT-->", content)


def build(notes_path, output=None):
    notes_path = (ROOT / notes_path).resolve()
    notes = read(notes_path)
    report, rows, summary, selected = validate_notes(notes)
    output = Path(output) if output else report / "analysis-final-v1"
    code_hashes = {p.name: sha(p) for p in sorted(Path(__file__).parent.iterdir()) if p.suffix in (".py", ".html")}
    packages = {n: importlib.metadata.version(n) for n in ("numpy", "scipy", "Pillow", "transformers")}
    binding = {"notes_sha256": sha(notes_path), "code_sha256": code_hashes, "packages": packages}
    if output.exists():
        receipt = read(output / "receipt.json")
        if receipt["binding"] != binding or any(sha(output / p) != h for p, h in receipt["files_sha256"].items()):
            raise ValueError("analysis_output_changed_use_new_--output")
        return {"output": str(output), "resumed": True, "cases": len(selected)}
    tasks = {t["task_id"]: t for t in read(ROOT / notes["cohort"])["tasks"]}
    qwen = {r["task_id"]: r for r in read(ROOT / notes["qwen_index"])["entries"]}
    pairs = {r["task_id"]: r for r in read(ROOT / notes["sgg_results"])}
    authors = {r["task_id"]: r for r in read(ROOT / notes["authoring"])["tasks"]}
    graph_scores = {r["task_id"]: r["scores"] for r in read(ROOT / notes["graph_results"])["rows"]}
    labels = read(report / "labels.json")
    defaults = labels["normalization"]["defaulted_fields"]
    common = [r for r in rows if all(r.get(k) is not None for k in METRICS)]
    pack = {
        "schema": "level3-final-analysis-v1",
        "provenance": binding,
        "inputs_sha256": notes["inputs_sha256"],
        "human_labels_used_for_tuning": False,
        "assistant_visual_analysis_is_ground_truth": False,
        "summary": summary,
        "bootstrap": bootstrap(rows),
        "labels": {
            "all": label_counts(rows),
            "common": label_counts(common),
            "abstained": label_counts([r for r in rows if r not in common]),
            "user_defaulted_fields": sum(len(v) for v in defaults.values()),
            "user_defaulted_cases": len(defaults),
            "confirmation": labels["recheck"],
        },
        "cases": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".analysis-", dir=output.parent) as tmp:
        dest = Path(tmp)
        for note, candidate in zip(notes["cases"], selected, strict=True):
            tid = note["task_id"]
            result = qwen[tid]["result"]
            graphs, media, hashes = {}, {}, {}
            for role in ("source", "target"):
                g = pairs[tid]["roles"][role]
                if sha(g["graph_path"]) != g["graph_sha256"]:
                    raise ValueError("graph_changed:" + tid)
                rec = read(g["graph_path"])
                graphs[role] = {k: rec[k] for k in ("image", "source_graph", "grounding")}
                media[role] = copy_media(
                    result[role + "_path"],
                    dest,
                    result[role + "_sha256"] if role == "source" else result["target"]["sha256"],
                )
                hashes[role] = {"image_sha256": sha(result[role + "_path"]), "graph_sha256": g["graph_sha256"]}
            case = {
                **note,
                "candidate": {k: v for k, v in candidate.items() if k not in ("qualitative_analysis", "status")},
                "instruction": tasks[tid]["instruction_en"],
                "media": media,
                "graphs": graphs,
                "hashes": hashes,
                "grounding": authors[tid]["grounding"],
                "scoring_evidence": graph_scores[tid],
                "status": "assistant_visually_inspected",
            }
            if note.get("inspect_clip_preprocessing"):
                name = f'media/clip-input-{note["index"]}.png'
                case["clip_processor"] = clip_preview(result["target_path"], dest / name)
                case["clip_preview"] = name
            pack["cases"].append(case)
        write(dest / "analysis.json", pack)
        shutil.copyfile(report / "results.csv", dest / "results.csv")
        (dest / "index.html").write_text(render(pack), encoding="utf-8")
        write(
            dest / "receipt.json",
            {
                "binding": binding,
                "files_sha256": {str(p.relative_to(dest)): sha(p) for p in sorted(dest.rglob("*")) if p.is_file()},
            },
        )
        dest.rename(output)
    return {"output": str(output), "resumed": False, "cases": len(selected)}
