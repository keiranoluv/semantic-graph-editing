"""Reuse the existing reviewer, with truthful method/scope metadata and new paths."""

import json
from http.server import ThreadingHTTPServer

from src.datasets.gqa_full import atomic_write
from src.knowledge.transe.core import sha256
from tools.prepare_magicbrush import ROOT, digest, encode, fixed
from tools.review_magicbrush import TEMPLATES, handler, make_packet


def review_dir(config):
    path = (ROOT / config["reviews"]).resolve()
    if not path.is_relative_to(ROOT / "reviews"):
        raise ValueError("Human review outputs must live under reviews/")
    return path


def build(config, output):
    pointer = json.loads((output / "current-compilation.json").read_text())
    edits_path = (output / pointer["file"]).resolve()
    if edits_path.parent != output.resolve() or sha256(edits_path) != pointer["sha256"]:
        raise ValueError("Compilation pointer is invalid")
    packet = make_packet(output / "manifest.json", edits_path)
    manifest = json.loads((output / "manifest.json").read_text())
    packet.update(
        selection=manifest["selection"],
        graph_origin="BLIP_source_caption_DINO_boxes_CLIP_checks_source_anchored_spatial_pseudo_labels_not_GT",
        graph_scope="PARTIAL object graph; left_of/right_of/above/below mean bbox-center order only. No semantic support/action edges.",
        automatic_approval=False,
        training_enabled=False,
        review_interpretation="Fixed grammar-supported pilot, not an unbiased estimate of full-dataset annotation accuracy. Review failed cases too; no replacements.",
    )
    packet.pop("packet_sha256")
    packet["packet_sha256"] = digest(packet)
    destination = review_dir(config)
    fixed(destination / "packets" / f"{packet['packet_sha256']}.json", encode(packet))
    data = json.dumps(packet, ensure_ascii=False).replace("<", "\\u003c").replace("&", "\\u0026")
    page = (TEMPLATES / "index.html").read_text().replace("__REVIEW_DATA__", data)
    page = page.replace("MagicBrush / Đồ thị & thao tác", "MagicBrush / Grounded anchor · pilot v2")
    page = page.replace(
        "Ảnh từ MagicBrush. Khung vàng, khi có, là bounding box do model dự đoán, không phải box GT.",
        "Ảnh thật từ MagicBrush. Khung vàng là box dự đoán/anchor, không phải GT. "
        "Ở ảnh đích, object ngữ cảnh và đổi màu giữ vùng neo của ảnh nguồn sau kiểm tra ổn định.",
    )
    page = page.replace(
        "Mẫu được khóa trước khi sinh graph. Các graph là nhãn dự đoán; cần kiểm tra cả lỗi bỏ sót.",
        "20 yêu cầu thuộc phạm vi hỗ trợ, chọn trước inference. Graph cục bộ: object và vị trí tâm box. "
        "Chưa biểu diễn quan hệ hành động/hỗ trợ. Chấm cả lỗi bỏ sót chỉnh sửa trong phạm vi; kiểm cả các ca bị bộ lọc loại.",
    )
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "index.html").write_text(page)
    atomic_write(destination / "review-data.json", packet)
    for name in ("viewer.js", "viewer.css"):
        content = (TEMPLATES / name).read_text()
        if name == "viewer.js":
            content = content.replace(
                '"CHỜ SINH ĐỒ THỊ: ảnh thật đã sẵn sàng. Chưa thể chấm chất lượng graph/alignment/thao tác. Không tính đây là cặp nhãn đã hoàn thành."',
                '`CHƯA CÓ GRAPH ĐỂ CHẤM: ${(row.label?.problems||[]).join("; ")}. Giữ lại ảnh; không tính đây là nhãn đạt chất lượng.`',
            )
        (destination / name).write_text(content)
    print(
        f"[READY] {len(packet['cases'])} pilot pairs; {sum(r['graphs_ready'] for r in packet['cases'])} with image evidence: {destination/'index.html'}"
    )
    return packet


def serve(config, port):
    output = review_dir(config)
    packet = json.loads((output / "review-data.json").read_text())
    if digest({k: v for k, v in packet.items() if k != "packet_sha256"}) != packet["packet_sha256"]:
        raise ValueError("Review packet was changed")
    with ThreadingHTTPServer(("127.0.0.1", port), handler(output, packet)) as server:
        print(f"Grounded pilot review: http://127.0.0.1:{server.server_port}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


def audit(config, output, csv_path):
    from tools.edit_model_v1.data import review_gate

    directory = review_dir(config)
    packet_path = directory / "review-data.json"
    packet = json.loads(packet_path.read_text())
    pointer = json.loads((output / "current-compilation.json").read_text())
    edits_path = (output / pointer["file"]).resolve()
    if (
        edits_path.parent != output.resolve()
        or sha256(edits_path) != pointer["sha256"]
        or packet["edits_file_sha256"] != pointer["sha256"]
    ):
        raise ValueError("Reviews are for another compilation; rebuild the viewer, do not reuse old decisions")
    gate = review_gate(
        output / "manifest.json", edits_path, packet_path, csv_path, minimum=20, minimum_correct_fraction=0.9
    )
    # This is a diagnostic pilot gate only. It cannot enable full inference or
    # training, nor certify the reliability of the unprocessed dataset.
    result = {
        "pilot_quality_gate_passed": gate.pop("ready"),
        **gate,
        "ready_for_training": False,
        "automatic_full_inference": False,
        "scope": "fixed20 grammar-supported train pilot; not full-population accuracy",
        "all_20_required": True,
        "uncertain_is_not_correct": True,
    }
    fixed(directory / "audits" / f"audit-{digest(result)[:16]}.json", encode(result))
    return result
