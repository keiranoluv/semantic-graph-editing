"""CPU-only failure audit: distinguish unassigned labels from verified no-ops.

Reads frozen v2 evidence. Never changes graphs, reviews, config or thresholds;
never generates target edits from an instruction or starts model inference.
"""

import argparse
import html
import json
from collections import Counter
from pathlib import Path

from src.knowledge.transe.core import sha256
from tools.magicbrush_graphs import iou
from tools.magicbrush_grounded_v1.policy import color, cosine
from tools.magicbrush_grounded_v2.runner import identity, key, read_cache
from tools.prepare_magicbrush import ROOT, digest, encode, fixed

CASES = {
    "magicbrush:73536:1": (
        "replace_node",
        "Tree → piñata",
        "Ảnh đích có vật trang trí piñata thay vùng cây; detector vẫn gọi một vùng là christmas tree.",
    ),
    "magicbrush:128175:2": (
        "add_node",
        "Add a pear",
        "Có thêm quả lê ở phía trên bên trái; detection pear cũ vẫn còn. Không đủ căn cứ gọi mọi detection pear là lê thật.",
    ),
    "magicbrush:349918:1": (
        "replace_node",
        "Chair → yellow",
        "Cùng chiếc ghế có phần đệm đổi từ tối sang vàng; không cần đổi identity chỉ vì màu thay đổi.",
    ),
    "magicbrush:423742:1": (
        "replace_node",
        "Cat → hedgehog",
        "Vùng con mèo bên phải laptop được thay bằng con nhím nhỏ hơn; bbox thay kích thước là điều có thể xảy ra.",
    ),
    "magicbrush:198947:1": (
        "replace_node",
        "Dog → balloon",
        "Con chó được thay bằng hình chó làm bằng bóng bay; class/material cần mô tả rõ trong ontology, không tự suy ra thuộc tính từ câu yêu cầu.",
    ),
}


def summarize(entries):
    """An empty rejected transaction is not a verified unchanged target graph."""
    return {
        "pilot_pairs": len(entries),
        "status_counts": dict(Counter(r["status"] for r in entries)),
        "nonempty_operation_pairs": sum(bool(r.get("operations")) for r in entries),
        "empty_operation_pairs": sum("operations" in r and not r["operations"] for r in entries),
        "pairs_without_graph_transaction": sum("operations" not in r for r in entries),
        "passed_automatic_filters": sum(r.get("passed_automatic_filters") is True for r in entries),
        "problem_counts": dict(Counter(p for r in entries for p in r.get("problems", []))),
        "verified_noop_pairs": 0,
        "interpretation": "Rejected empty transactions and missing graphs are unassigned labels, not verified no-op answers. No no-op verification exists in this recipe.",
    }


def measurements(cached, intent, config):
    left, right = cached["source"], cached["target"]
    before, after = intent.get("before", {}).get("label"), intent.get("after", {}).get("label")
    a = [n for n in left["nodes"] if n["label"] == (before or after)]
    b = [n for n in right["nodes"] if n["label"] == (after or before)]
    result = {
        "focus_source_detection_count": len(a),
        "focus_target_detection_count": len(b),
        "source_presence_scores": left["presence_scores"],
        "target_presence_scores": right["presence_scores"],
        "source_detection_labels": [n["label"] for n in left["nodes"]],
        "target_detection_labels": [n["label"] for n in right["nodes"]],
    }
    if len(a) == len(b) == 1:
        result.update(
            focus_iou=iou(a[0]["bbox"], b[0]["bbox"]),
            focus_crop_cosine=cosine(a[0]["embedding"], b[0]["embedding"]),
            source_color_assignment=color(a[0], config),
            target_color_assignment=color(b[0], config),
        )
    if len(b) == 1:
        result["target_focus_anchor_proof"] = cached["evidence"]["target_anchors"][b[0]["id"]]
    return result


def photo(meta, proposal, role):
    path = ROOT / meta["path"]
    if sha256(path) != meta["sha256"]:
        raise ValueError("Diagnostic image differs from frozen evidence")
    shapes = []
    for n in proposal["nodes"]:
        a, b, c, d = n["bbox"]
        caption = html.escape(f"{n['id']} {n['label']} {n['detector_score']:.3f}")
        shapes.append(
            f'<rect x="{a}" y="{b}" width="{c-a}" height="{d-b}"/><text x="{a+4}" y="{max(b+20,20)}">{caption}</text>'
        )
    # The URI references the original image bytes; this tool creates no image evidence.
    return (
        f'<div><h4>{role}</h4><div class="photo"><img src="../../{html.escape(meta["path"], quote=True)}">'
        f'<svg viewBox="0 0 1000 1000" preserveAspectRatio="none">{"".join(shapes)}</svg></div></div>'
    )


def audit(directory, output, viewer):
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    run_identity = json.loads((directory / "inference-identity.json").read_text())
    config = run_identity["config"]
    if {k: v for k, v in run_identity.items() if k != "runtime"} != identity(config, directory):
        raise ValueError("Frozen v2 inference source/config/manifest differs")
    pointer = json.loads((directory / "current-compilation.json").read_text())
    edits_path = (directory / pointer["file"]).resolve()
    if edits_path.parent != directory.resolve() or sha256(edits_path) != pointer["sha256"]:
        raise ValueError("Compilation pointer differs")
    edits = json.loads(edits_path.read_text())
    if edits["dataset_manifest_sha256"] != sha256(manifest_path):
        raise ValueError("Label compilation belongs to another manifest")
    selected = manifest["review_ids"]
    labels = {r["id"]: r for r in edits["records"]}
    rows = {r["id"]: r for r in manifest["records"]}
    if len(selected) != 20 or any(
        rows[k]["split"] != "train" or rows[k]["official_split"] != "train" for k in selected
    ):
        raise ValueError("Audit is restricted to the frozen 20 train pairs")
    diagnostics, cards = [], []
    for identifier in selected:
        row, label = rows[identifier], labels[identifier]
        path = directory / "evidence" / f"{key(identifier)}.json"
        cached = read_cache(path, digest(run_identity), identifier)
        if label.get("evidence_sha256") != sha256(path):
            raise ValueError("Compiled per-pair evidence hash differs")
        facts = measurements(cached, label.get("scope", {}), config)
        state = (
            "unassigned" if label["status"] != "candidate" or not label.get("operations") else "unverified_candidate"
        )
        entry = {
            "id": identifier,
            "instruction": row["instruction"],
            "label_state": state,
            "historical_status": label["status"],
            "historical_operation_count": len(label.get("operations", [])),
            "problems": label.get("problems", []),
            "measurements": facts,
            "evidence_sha256": sha256(path),
        }
        if identifier in CASES:
            expected, title, observation = CASES[identifier]
            entry["qualitative_feedback"] = {
                "source": "user reported incorrect missing edit; assistant inspected both original images",
                "expected_main_edit_family": expected,
                "case": title,
                "visual_observation": observation,
                "is_human_review_csv": False,
                "is_approved_ground_truth": False,
            }
        diagnostics.append(entry)
        esc = html.escape
        pictures = photo(row["source_image"], cached["source"], "Detection ảnh nguồn") + photo(
            row["target_image"], cached["target"], "Detection ảnh đích"
        )
        verdict = (
            "CHƯA GÁN ĐƯỢC NHÃN ĐÍCH. Không diễn giải operations=[] thành “ảnh không thay đổi”."
            if state == "unassigned"
            else "Candidate chưa được người kiểm; không tự approved."
        )
        cards.append(
            f'<article id="{esc(identifier,quote=True)}"><h2>{esc(identifier)} — {esc(row["instruction"])}</h2>'
            f'<p class="bad">{verdict}</p>'
            f'<div class="pair">{pictures}</div><p>Box và tên là dự đoán DINO, có thể sai. Đây là detection độc lập; không dùng bản sao graph nguồn làm đáp án ảnh đích.</p>'
            f"<details><summary>Số đo và lý do từ chối</summary><pre>{esc(json.dumps(entry,ensure_ascii=False,indent=2))}</pre></details></article>"
        )
    summary = summarize([labels[k] for k in selected])
    report = {
        "schema_version": 1,
        "GPU_started": False,
        "training_started": False,
        "summary": summary,
        "user_qualitative_failed_cases": 5,
        "saved_human_review_csv_created": False,
        "full_dataset_accuracy_estimated": False,
        "ready_for_training": False,
        "pdf_requirements": {
            "file_sha256": sha256(ROOT / "Project01_Graph_Editing.pdf"),
            "pages": [2, 3, 4, 5],
            "minimum_pairs": 300,
            "minimum_human_checks": 20,
            "must_represent_requested_node_edge_edits": True,
        },
        "input_sha256": {
            str(p.relative_to(ROOT)): sha256(p)
            for p in (manifest_path, edits_path, directory / "inference-identity.json")
        },
        "records": diagnostics,
    }
    output = output.resolve()
    viewer = viewer.resolve()
    if not output.is_relative_to(ROOT / "reports") or viewer.parent != ROOT / "reviews":
        raise ValueError("Write audit under reports/ and diagnostic viewer one directory under reviews/")
    fixed(output / "audit.json", encode(report))
    page = (
        '<!doctype html><html lang="vi"><meta charset="utf-8"><title>MagicBrush — chẩn đoán nhãn bị từ chối</title>'
        "<style>body{font:16px system-ui;max-width:1300px;margin:25px auto;padding:15px;background:#f4f5f8}article{background:white;padding:20px;margin:20px 0}.bad{color:#9b1717;font-weight:bold}.pair{display:grid;grid-template-columns:1fr 1fr;gap:15px}.photo{position:relative}.photo img{width:100%;display:block}.photo svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}rect{fill:none;stroke:#ffdf00;stroke-width:3}text{fill:#ffdf00;stroke:#000;stroke-width:2;paint-order:stroke;font-size:20px}pre{white-space:pre-wrap;overflow-wrap:anywhere}@media(max-width:650px){.pair{grid-template-columns:1fr}}</style>"
        "<h1>Chẩn đoán pilot — quan sát và nhãn đích chưa xác minh</h1>"
        f'<p>{summary["pilot_pairs"]} cặp train; {summary["nonempty_operation_pairs"]} cặp có thao tác; {summary["passed_automatic_filters"]} candidate qua bộ lọc.</p>'
        "<p>Giữ nguyên bằng chứng cũ. Không tạo nhãn hoặc đánh giá approved. Trang này hiển thị các detection thật đã lưu để tìm lỗi pipeline; không phải nhãn đúng để train.</p>"
        + "".join(cards)
        + "</html>"
    )
    fixed(viewer / "index.html", page.encode())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/level2/magicbrush-grounded-anchor-v2")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/level2/magicbrush-grounded-failure-audit-v1")
    parser.add_argument("--viewer", type=Path, default=ROOT / "reviews/magicbrush-grounded-diagnostics-v1")
    args = parser.parse_args(argv)
    audit(args.data.resolve(), args.output, args.viewer)


if __name__ == "__main__":
    main()
