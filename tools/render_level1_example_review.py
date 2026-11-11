"""Render local JPEGs and GT boxes for manual PredCls hit/miss review."""

import argparse
import html
import json
import os
from pathlib import Path


def box_style(box: list[float], width: int, height: int) -> str:
    x1, y1, x2, y2 = box
    return (
        f"left:{100 * x1 / width:.4f}%;top:{100 * y1 / height:.4f}%;"
        f"width:{100 * (x2 - x1) / width:.4f}%;height:{100 * (y2 - y1) / height:.4f}%"
    )


def render_case(row: dict, kind: str, number: int, metadata: dict, images_dir: Path, output: Path) -> str:
    image_id = str(row["image_id"])
    if not image_id.isdigit():
        raise ValueError(f"Unsafe image ID: {image_id}")
    info = metadata[image_id]
    width, height = int(info["width"]), int(info["height"])
    if min(width, height) <= 0:
        raise ValueError(f"Invalid image dimensions: {image_id}")
    image_path = images_dir / f"{image_id}.jpg"
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    image_src = html.escape(os.path.relpath(image_path, output.parent), quote=True)
    subject = html.escape(row["subject_class"])
    obj = html.escape(row["object_class"])
    gt = html.escape(row["gt_predicate"])
    predicted = html.escape(row["predicted_predicate"])
    subject_box = box_style(row["subject_box_xyxy"], width, height)
    object_box = box_style(row["object_box_xyxy"], width, height)
    reason = html.escape(row["reason"])
    label = "Hit" if kind == "hit_at_50" else "Miss"
    return f"""
<article class="card {kind}">
  <h2>{label} {number} · image {image_id}</h2>
  <p><strong>GT:</strong> {subject} → <b>{gt}</b> → {obj}<br>
     <strong>Model:</strong> {subject} → <b>{predicted}</b> → {obj}<br>
     <strong>Pair rank:</strong> {int(row['predicted_rank'])};
     <strong>score:</strong> {float(row['predicted_pair_score']):.3f};
     <strong>metric reason:</strong> {reason}</p>
  <div class="photo" style="aspect-ratio:{width}/{height}">
    <img src="{image_src}" alt="VG150 validation image {image_id}">
    <div class="box subject" style="{subject_box}"><span>Subject: {subject}</span></div>
    <div class="box object" style="{object_box}"><span>Object: {obj}</span></div>
  </div>
  <details><summary>Box coordinates</summary><pre>subject: {html.escape(str(row['subject_box_xyxy']))}\nobject: {html.escape(str(row['object_box_xyxy']))}</pre></details>
</article>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, default=Path("data/raw/vg150/images"))
    parser.add_argument("--image-metadata", type=Path, default=Path("data/raw/vg150/image_data.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = json.loads(args.examples.read_text(encoding="utf-8"))
    if report.get("split") != "val" or report.get("protocol") != "PredCls_graph_constrained_top1_per_pair_R@50":
        raise ValueError("Expected VG150 validation R@50 review candidates")
    metadata = {str(item["image_id"]): item for item in json.loads(args.image_metadata.read_text(encoding="utf-8"))}
    cards = []
    for kind in ("hit_at_50", "missed_at_50"):
        rows = report["examples"][kind]
        cards.extend(
            render_case(row, kind, number, metadata, args.images_dir, args.output) for number, row in enumerate(rows, 1)
        )
    page = f"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>VG150 PredCls — review 5 hit / 5 miss</title>
<style>
body{{font:16px/1.5 system-ui,sans-serif;max-width:1120px;margin:0 auto;padding:24px;color:#18202b;background:#f4f6f9}}
h1{{font-size:1.7rem}}.note{{background:#fff4d5;padding:12px;border-radius:8px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,480px),1fr));gap:16px}}
.card{{background:white;padding:16px;border-radius:10px;box-shadow:0 1px 5px #0002}}
.card h2{{margin:0 0 8px;font-size:1.25rem}}.photo{{position:relative;width:100%;overflow:hidden;background:#ddd}}
.photo img{{display:block;width:100%;height:100%}}.box{{position:absolute;box-sizing:border-box;border:3px solid;pointer-events:none}}
.box span{{position:absolute;top:0;left:0;color:white;font-size:12px;font-weight:700;padding:2px 4px;white-space:nowrap}}
.subject{{border-color:#ff3b30}}.subject span{{background:#c5241c}}.object{{border-color:#00a65a}}.object span{{background:#087c45}}
.hit_at_50{{border-top:5px solid #087c45}}.missed_at_50{{border-top:5px solid #bd6715}}
pre{{white-space:pre-wrap}}
</style></head><body>
<h1>VG150 PredCls · 5 hit / 5 miss R@50</h1>
<p class="note">Viền đỏ: subject; viền xanh: object. “Hit” là triplet GT có trong top 50 dự đoán;
“miss” là triplet GT vắng mặt. Đây là nhãn theo metric: hãy xem ảnh và box trước khi kết luận dự đoán sai về nghĩa.</p>
<p>Checkpoint: {html.escape(str(report['checkpoint']))} · split: val · pool: {int(report['image_pool_size'])} ảnh</p>
<div class="grid">{''.join(cards)}</div>
</body></html>"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(page, encoding="utf-8")
    print(f"[OK] {len(cards)} review cards: {args.output}")


if __name__ == "__main__":
    main()
