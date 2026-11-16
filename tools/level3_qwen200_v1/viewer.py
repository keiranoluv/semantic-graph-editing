"""Instruction preview and the existing two-field Qwen-only review interface."""

import csv
import html
from pathlib import Path

from tools.level3_image_graph_v1.core import ROOT, digest, read, write
from tools.level3_qwen100_v1.viewer import thumb

from .core import json_data


def render(rows, out, authoring=False):
    out = Path(out)
    if authoring:
        prior = {t["task_id"]: t for t in read(ROOT / "configs/level3/qwen100-instructions-v1.json")["tasks"]}
        cards = []
        for r in rows:
            old = prior[r["parent_task_id"]]
            cards.append(
                f'<article id="request-{r["request_index"]}"><h2>{r["request_index"]} · {html.escape(r["image_id"])}</h2>'
                f'<img src="{thumb(r["source_path"])}" alt="Ảnh nguồn">'
                f'<p><b>Câu cũ:</b> {html.escape(old["instruction_en"])}</p>'
                f'<p><b>Câu mới:</b> {html.escape(r["instruction_en"])}</p>'
                f'<p>Loại: {html.escape(r["operation"]["kind"])} · {html.escape(r["evaluation_group"])}</p></article>'
            )
        page = """<!doctype html><html lang="vi"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>100 yêu cầu mới · Qwen</title>
<style>body{font:16px system-ui;max-width:1000px;margin:24px auto;padding:0 16px;background:#f3f5f7;color:#203344}
article{background:white;padding:20px;margin:20px 0;border-radius:10px}img{max-width:100%;max-height:480px}p{line-height:1.6}</style>
<h1>Yêu cầu 101–200 · mỗi câu chạy từ ảnh nguồn gốc</h1>
<p>100 câu mới: 40 thêm, 38 xóa, 12 thay thế, 10 đổi màu. Câu cũ chỉ hiển thị để đối chiếu; launcher không chạy lại.
Đây là bản soạn instruction của assistant, chưa có ảnh đích hay nhãn chất lượng cho lượt mới.</p>"""
        (out / "instructions.html").write_text(page + "\n".join(cards) + "</html>", encoding="utf-8")
        with (out / "instructions.csv").open("x", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["request_index", "task_id", "image_id", "dataset", "kind", "old_instruction", "new_instruction"]
            )
            for r in rows:
                writer.writerow(
                    [
                        r["request_index"],
                        r["task_id"],
                        r["image_id"],
                        r["dataset"],
                        r["operation"]["kind"],
                        prior[r["parent_task_id"]]["instruction_en"],
                        r["instruction_en"],
                    ]
                )
        return
    cases, reviews = [], []
    for row in rows:
        available = row["status"] == "ok" and bool(row.get("target_path"))
        cases.append(
            {
                "task_id": row["task_id"],
                "prompt": row["prompt"],
                "source": thumb(row["source_path"]),
                "target": thumb(row["target_path"]) if available else None,
            }
        )
        reviews.append(
            {
                "task_id": row["task_id"],
                "model": "qwen21",
                "status": "pending",
                "edit_fulfillment": None,
                "preservation": None,
            }
        )
    template = {
        "schema": "qwen-only-review-v2",
        "results_sha256": digest(rows),
        "review_policy": "two_manual_fields; no_notes_or_named_reviewer_required",
        "reviews": reviews,
    }
    write(out / "review-template.json", template)
    data = json_data({"cases": cases, "review": template, "download": "qwen200-extension-reviewed-v1.json"})
    source = ROOT / "tools/level3_qwen_review_v2.html"
    (out / "comparison.html").write_text(source.read_text().replace("__REVIEW_DATA__", data), encoding="utf-8")
