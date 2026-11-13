"""Existing review viewer and five-axis CSV gate, with truthful joint-VLM metadata."""

import json
from http.server import ThreadingHTTPServer
from pathlib import Path

from src.knowledge.transe.core import sha256
from tools.edit_model_v1.data import review_gate
from tools.magicbrush_joint_v1.prepare import check_run
from tools.magicbrush_joint_v1.storage import read, write_once
from tools.prepare_magicbrush import ROOT, digest
from tools.review_magicbrush import TEMPLATES, handler, make_packet


def compilation_path(cfg, path=None):
    output = ROOT / cfg["output"]
    if path is None:
        pointer = read(output / "current-compilation.json")
        path = ROOT / pointer["path"]
        if sha256(path) != pointer["sha256"]:
            raise ValueError("compilation_pointer_changed")
    path = Path(path).resolve()
    if path.parent != (output / "compilations").resolve() or path.stem != digest(read(path)):
        raise ValueError("invalid_compilation_evidence")
    run, _ = check_run(cfg)
    if read(path)["run_sha256"] != digest(run):
        raise ValueError("compilation_run_mismatch")
    return path


def directory(cfg, path):
    return ROOT / cfg["reviews"] / path.stem


def build(cfg, path=None):
    path = compilation_path(cfg, path)
    output = ROOT / cfg["output"]
    packet = make_packet(output / "manifest.json", path)
    packet.update(
        selection=read(output / "run.json")["selection"],
        graph_origin="joint_local_VLM_two_images_plus_instruction_observations_not_GT",
        graph_scope="edit-focused partial object/attribute/relation graphs, not exhaustive scene graphs",
        automatic_approval=False,
        training_enabled=False,
        review_interpretation="Intent-stratified train pilot, not an unbiased dataset accuracy estimate. Review failed cases; do not replace them. Missing/ambiguous graphs cannot pass the existing five-axis gate.",
    )
    packet.pop("packet_sha256")
    packet["packet_sha256"] = digest(packet)
    destination = directory(cfg, path)
    write_once(destination / "packets" / f"{packet['packet_sha256']}.json", packet)
    write_once(destination / "review-data.json", packet)
    data = json.dumps(packet, ensure_ascii=False).replace("<", "\\u003c").replace("&", "\\u0026")
    page = (TEMPLATES / "index.html").read_text().replace("__REVIEW_DATA__", data)
    page = page.replace("MagicBrush / Đồ thị & thao tác", "MagicBrush / Joint VLM · pilot v1")
    page = page.replace(
        "Mẫu được khóa trước khi sinh graph. Các graph là nhãn dự đoán; cần kiểm tra cả lỗi bỏ sót.",
        "20 cặp train chọn theo nhóm yêu cầu và hash trước inference. Graph tập trung chỉnh sửa, không đầy đủ toàn cảnh. Kiểm cả lỗi bỏ sót và thay đổi ngoài yêu cầu.",
    )
    # Immutable per-compilation viewers: no old packet, decision or artifact is overwritten.
    for name, content in (
        ("index.html", page),
        ("viewer.css", (TEMPLATES / "viewer.css").read_text()),
        (
            "viewer.js",
            (TEMPLATES / "viewer.js")
            .read_text()
            .replace(
                '"CHỜ SINH ĐỒ THỊ: ảnh thật đã sẵn sàng. Chưa thể chấm chất lượng graph/alignment/thao tác. Không tính đây là cặp nhãn đã hoàn thành."',
                '`CHƯA CÓ GRAPH HỢP LỆ: ${(row.label?.problems||[]).join("; ")}. Xem chi tiết lỗi; mẫu này chưa qua cổng review.`',
            ),
        ),
    ):
        target = destination / name
        if target.exists() and target.read_text() != content:
            raise ValueError("review_viewer_changed")
        if not target.exists():
            target.write_text(content)
    return {
        "review_directory": str(destination),
        "cases": len(packet["cases"]),
        "packet_sha256": packet["packet_sha256"],
    }


def serve(cfg, path=None, port=8771):
    path = compilation_path(cfg, path)
    destination = directory(cfg, path)
    packet = read(destination / "review-data.json")
    if digest({k: v for k, v in packet.items() if k != "packet_sha256"}) != packet["packet_sha256"] or packet[
        "edits_file_sha256"
    ] != sha256(path):
        raise ValueError("review_packet_changed")
    with ThreadingHTTPServer(("127.0.0.1", port), handler(destination, packet)) as server:
        print(f"Review: http://127.0.0.1:{server.server_port}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


def audit(cfg, csv_path, path=None):
    path = compilation_path(cfg, path)
    packet_compilation = path
    if read(path)["scope"] == "expanded":
        expansion = read(ROOT / cfg["output"] / "expansion.json")
        packet_compilation = compilation_path(cfg, expansion["compilation_path"])
    return review_gate(
        ROOT / cfg["output"] / "manifest.json",
        path,
        directory(cfg, packet_compilation) / "review-data.json",
        csv_path,
        minimum=20,
        minimum_correct_fraction=cfg["minimum_review_fraction"],
    )


def expand(cfg, csv_path, path=None):
    path = compilation_path(cfg, path)
    if read(path)["scope"] != "pilot":
        raise ValueError("expansion_requires_pilot_evidence")
    gate = audit(cfg, csv_path, path)
    if not gate["ready"]:
        raise ValueError(f"human_quality_gate_not_met:{gate}")
    run, manifest = check_run(cfg)
    result = {
        "run_sha256": digest(run),
        "review_gate": gate,
        "csv_path": str(Path(csv_path).resolve()),
        "compilation_path": str(path),
        "ids": sorted(manifest["splits"]["train"] + manifest["splits"]["val"]),
        "scope": "project_train_and_val_only; official_train; never_project_test",
        "automatic_gpu_launch": False,
    }
    write_once(ROOT / cfg["output"] / "expansion.json", result)
    return result
