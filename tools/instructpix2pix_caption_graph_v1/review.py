"""Existing five-axis viewer/CSV format, with caption evidence and honest denominators."""

import csv
import hashlib
import json
from collections import Counter
from http.server import ThreadingHTTPServer
from pathlib import Path

from src.editing.operations import EditGraph, Operation, apply_operations
from src.knowledge.transe.core import sha256
from tools.magicbrush_joint_v1.storage import read, write_once
from tools.prepare_magicbrush import ROOT, digest, encode, fixed
from tools.review_magicbrush import AXES, FIELDS, TEMPLATES, export_reviews, handler, validate_reviews


def evidence(row):
    return digest(
        {
            k: row[k]
            for k in (
                "id",
                "source_image",
                "target_image",
                "original_prompt",
                "edited_prompt",
                "instruction",
                "provenance",
                "label",
            )
        }
    )


def verify_images(packet):
    for row in packet["cases"]:
        for role in ("source_image", "target_image"):
            image = row[role]
            if image.get("status") == "unavailable":
                continue
            if sha256(ROOT / image["path"]) != image["sha256"]:
                raise ValueError("review_image_bytes_changed")
        if row["evidence_sha256"] != evidence(row):
            raise ValueError("per_sample_review_evidence_changed")


def default_directory(path):
    return ROOT / "reviews/instructpix2pix-caption-graph-v1" / path.name


def build(output, compilation=None, directory=None):
    from tools.instructpix2pix_caption_graph_v1.runner import check_compilation, image_receipts

    path, manifest, edits = check_compilation(output, compilation)
    directory = Path(directory) if directory else default_directory(path)
    if not directory.resolve().is_relative_to(ROOT / "reviews"):
        raise ValueError("review_directory_must_be_under_reviews")
    labels = {r["id"]: r for r in edits["records"]}
    rows = {r["id"]: r for r in manifest["records"]}
    candidates = sorted(
        (r["id"] for r in edits["records"] if r["split"] == "train" and r["status"] == "candidate"),
        key=lambda i: digest([manifest["run_sha256"], "conditional-review", i]),
    )[:20]
    selected = list(dict.fromkeys(manifest["review_ids"] + candidates))
    images = image_receipts(output)
    cases = []
    for identifier in selected:
        original = rows[identifier]
        label = labels[identifier]
        image = images.get(identifier, {})
        row = {
            **original,
            "instruction": original["edit_prompt"],
            "img_id": str(original["global_row"]),
            "turn_index": 0,
            "label": label,
            "graphs_ready": False,
            "reviewable": image.get("status") == "available",
            "cohorts": [
                name
                for name, ids in (("diagnostic", manifest["review_ids"]), ("accepted_conditional", candidates))
                if identifier in ids
            ],
        }
        for role in ("source_image", "target_image"):
            row[role] = image.get(
                role,
                {
                    "path": "",
                    "sha256": "unavailable",
                    "pixel_sha256": "unavailable",
                    "width": 0,
                    "height": 0,
                    "status": "unavailable",
                },
            )
        if label["stages"]["structural_validity"]:
            source, target = (EditGraph.from_dict(label[k]) for k in ("source_graph", "target_graph"))
            ops = tuple(Operation.from_dict(o) for o in label["operations"])
            if apply_operations(source, ops) != target:
                raise ValueError("review_transaction_changed")
            row["graphs_ready"] = True
        row["evidence_sha256"] = evidence(row)
        cases.append(row)
    packet = {
        "schema_version": 1,
        "dataset_manifest_sha256": sha256(path / "manifest.json"),
        "edits_file_sha256": sha256(path / "edits.json"),
        "run_sha256": manifest["run_sha256"],
        "cases": cases,
        "diagnostic_review_ids": manifest["review_ids"],
        "conditional_review_ids": candidates,
        "axes": AXES,
        "export_fields": FIELDS,
        "selection": "diagnostic SRS 20 of uniform diagnostic 100 before filters; independent uniform hash sample up to 20 accepted train candidates; overlap reviewed once",
        "graph_origin": "caption_derived_pseudo_labels_unverified_against_images",
        "human_reviews_created": 0,
        "automatic_approval": False,
    }
    packet["packet_sha256"] = digest(packet)
    verify_images(packet)
    write_once(directory / "packets" / f"{packet['packet_sha256']}.json", packet)
    write_once(directory / "review-data.json", packet)
    data = json.dumps(packet, ensure_ascii=False).replace("<", "\\u003c").replace("&", "\\u0026")
    page = (
        (TEMPLATES / "index.html")
        .read_text()
        .replace("__REVIEW_DATA__", data)
        .replace("MagicBrush", "InstructPix2Pix caption v1")
    )
    page = page.replace(
        '<h2 id="instruction"></h2>',
        '<h2 id="instruction"></h2><p><b>Caption nguồn:</b> <span id="original-caption"></span></p><p><b>Caption đích:</b> <span id="edited-caption"></span></p>',
    )
    page = page.replace(
        "Ảnh từ InstructPix2Pix caption v1. Khung vàng, khi có, là bounding box do model dự đoán, không phải box GT.",
        "Đối chiếu trực tiếp hai ảnh: caption chỉ mô tả ý định. Chấm Sai nếu thiếu graph hoặc thao tác không phản ánh khác biệt nhìn thấy.",
    )
    page = page.replace(
        "Mẫu được khóa trước khi sinh graph. Các graph là nhãn dự đoán; cần kiểm tra cả lỗi bỏ sót.",
        "20 mẫu chẩn đoán được chọn trước lọc; không thay mẫu lỗi. Có thêm mẫu candidate ngẫu nhiên để đo độ đúng có điều kiện. Chưa có ảnh thì chưa thể chấm.",
    )
    script = (TEMPLATES / "viewer.js").read_text()
    script = script.replace(
        '$("instruction").textContent=row.instruction;',
        '$("instruction").textContent=row.instruction;$("original-caption").textContent=row.original_prompt;$("edited-caption").textContent=row.edited_prompt;',
    )
    script = script.replace(
        'if(!row.graphs_ready)problems.push("chưa có graph để review");',
        'if(!row.reviewable)problems.push("chưa có đủ ảnh để kiểm chứng");if(r.decision==="correct"&&(!row.graphs_ready||row.label.status!=="candidate"))problems.push("mẫu không tạo được candidate không thể chấm Đúng");',
    )
    script = script.replace("select.disabled=!row.graphs_ready", "select.disabled=!row.reviewable").replace(
        '$("decision").disabled=!row.graphs_ready', '$("decision").disabled=!row.reviewable'
    )
    script = script.replace(
        'meta=row[role+"_image"];target.replaceChildren();',
        'meta=row[role+"_image"];target.replaceChildren();if(meta.status==="unavailable"){target.append(el("p","Ảnh chưa tải được; xem lỗi trong bằng chứng."));$(role+"-meta").textContent="unavailable";return;}',
    )
    script = script.replace(
        "CHỜ SINH ĐỒ THỊ: ảnh thật đã sẵn sàng. Chưa thể chấm chất lượng graph/alignment/thao tác. Không tính đây là cặp nhãn đã hoàn thành.",
        "KHÔNG CÓ GIAO DỊCH GRAPH HỢP LỆ. Xem lỗi parse/alignment trong chi tiết. Mẫu thất bại vẫn nằm trong mẫu số chẩn đoán.",
    )
    script = script.replace("magicbrush", "instructpix2pix-caption-v1")
    for name, content in (
        ("index.html", page),
        ("viewer.js", script),
        ("viewer.css", (TEMPLATES / "viewer.css").read_text()),
    ):
        fixed(directory / name, content.encode())
    # An empty CSV is a template, never fabricated human approval.
    fixed(directory / "review-template.csv", csv_bytes(packet, {}))
    return {
        "directory": str(directory),
        "cases": len(cases),
        "images_available": sum(r["reviewable"] for r in cases),
        "packet_sha256": packet["packet_sha256"],
        "human_reviews_created": 0,
    }


def validation_packet(packet):
    # Reuse the existing five-axis rules, allowing a reviewer with both images
    # to mark a failed parse as incorrect/uncertain (never correct).
    return {**packet, "cases": [{**r, "graphs_ready": r["reviewable"]} for r in packet["cases"]]}


def validate(packet, reviews):
    validate_reviews(validation_packet(packet), reviews)
    for row in packet["cases"]:
        if reviews.get(row["id"], {}).get("decision") == "correct" and (
            not row["graphs_ready"] or row["label"]["status"] != "candidate"
        ):
            raise ValueError("failed_candidate_cannot_be_human_correct")


def csv_bytes(packet, reviews):
    validate(packet, reviews)
    return export_reviews(validation_packet(packet), reviews)


def reliability(packet, reviews):
    validate(packet, reviews)

    def stats(ids):
        counts = Counter(reviews.get(i, {}).get("decision") or "unreviewed" for i in ids)
        determinate = counts["correct"] + counts["incorrect"]
        complete = bool(ids) and counts["unreviewed"] == 0
        return {
            "sample_size": len(ids),
            "decision_counts": dict(counts),
            "determinate_reviews": determinate,
            "correct_fraction_among_determinate": counts["correct"] / determinate if determinate else None,
            "correct_fraction_entire_preselected_cohort": counts["correct"] / len(ids) if complete else None,
            "incomplete_review": not complete,
        }

    diagnostic, conditional = stats(packet["diagnostic_review_ids"]), stats(packet["conditional_review_ids"])
    ready = (
        diagnostic["sample_size"] >= 20
        and not diagnostic["incomplete_review"]
        and conditional["determinate_reviews"] >= 20
        and conditional["correct_fraction_among_determinate"] >= 0.8
    )
    return {
        "ready": ready,
        "diagnostic_sample_wide": diagnostic,
        "accepted_candidate_conditional": conditional,
        "reviewed_pairs": sum(bool(r.get("decision")) for r in reviews.values()),
        "rejected_ids": sorted(i for i, r in reviews.items() if r.get("decision") in ("incorrect", "uncertain")),
        "review_packet_sha256": packet["packet_sha256"],
        "automatically_approved": False,
        "interpretation": "sample-wide yield and correctness conditional on auto-acceptance are separate; uncertainty is retained; no inter-rater agreement without independent second reviews",
    }


def checked_packet(output, compilation, directory):
    from tools.instructpix2pix_caption_graph_v1.runner import check_compilation, image_receipts

    path, manifest, _ = check_compilation(output, compilation)
    directory = Path(directory) if directory else default_directory(path)
    packet = read(directory / "review-data.json")
    if (
        packet["packet_sha256"] != digest({k: v for k, v in packet.items() if k != "packet_sha256"})
        or packet["dataset_manifest_sha256"] != sha256(path / "manifest.json")
        or packet["edits_file_sha256"] != sha256(path / "edits.json")
        or packet["diagnostic_review_ids"] != manifest["review_ids"]
    ):
        raise ValueError("review_packet_compilation_mismatch")
    # Rebuild expected per-case labels/metadata from current immutable inputs;
    # packet self-hash alone is insufficient evidence of their provenance.
    edits = read(path / "edits.json")
    labels = {r["id"]: r for r in edits["records"]}
    originals = {r["id"]: r for r in manifest["records"]}
    expected_conditional = sorted(
        (r["id"] for r in edits["records"] if r["split"] == "train" and r["status"] == "candidate"),
        key=lambda i: digest([manifest["run_sha256"], "conditional-review", i]),
    )[:20]
    expected_ids = set(manifest["review_ids"]) | set(expected_conditional)
    if (
        packet["conditional_review_ids"] != expected_conditional
        or len(packet["cases"]) != len(expected_ids)
        or {r["id"] for r in packet["cases"]} != expected_ids
    ):
        raise ValueError("review_selection_changed")
    receipts = image_receipts(output)
    for row in packet["cases"]:
        receipt = receipts.get(row["id"], {})
        if row["reviewable"] != (receipt.get("status") == "available") or row["graphs_ready"] != bool(
            labels[row["id"]]["stages"]["structural_validity"]
        ):
            raise ValueError("review_availability_changed")
        for role in ("source_image", "target_image"):
            expected_image = receipt.get(
                role,
                {
                    "path": "",
                    "sha256": "unavailable",
                    "pixel_sha256": "unavailable",
                    "width": 0,
                    "height": 0,
                    "status": "unavailable",
                },
            )
            if row[role] != expected_image:
                raise ValueError("review_image_receipt_changed")
        if row["instruction"] != originals[row["id"]]["edit_prompt"]:
            raise ValueError("review_instruction_changed")
        if row["label"] != labels[row["id"]] or any(
            row[k] != originals[row["id"]][k]
            for k in ("original_prompt", "edited_prompt", "edit_prompt", "provenance", "source_group_id", "split")
        ):
            raise ValueError("review_original_evidence_changed")
    verify_images(packet)
    return directory, packet


def audit(output, compilation=None, directory=None, csv_path=None):
    directory, packet = checked_packet(output, compilation, directory)
    if csv_path is None:
        return reliability(packet, {})
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(FIELDS):
            raise ValueError("unexpected_review_csv_schema")
        rows = list(reader)
    expected = {r["pair_id"]: r for r in csv.DictReader(csv_bytes(packet, {}).decode("utf-8-sig").splitlines())}
    if len(rows) != len(expected) or {r["pair_id"] for r in rows} != set(expected):
        raise ValueError("duplicate_missing_or_unknown_review_ids")
    reviews = {}
    for row in rows:
        fixed_fields = set(FIELDS) - {"decision", *AXES, "reviewer", "notes"}
        if any(row[k] != expected[row["pair_id"]][k] for k in fixed_fields):
            raise ValueError("review_csv_evidence_mismatch")
        reviews[row["pair_id"]] = {k: row[k] for k in ("decision", *AXES, "reviewer", "notes")}
    result = reliability(packet, reviews)
    result["review_csv_sha256"] = sha256(Path(csv_path))
    write_once(directory / "audits" / f"{digest(result)}.json", result)
    return result


def serve(output, compilation=None, directory=None, port=8774):
    directory, packet = checked_packet(output, compilation, directory)
    base = handler(directory, packet)

    class Handler(base):
        def do_POST(self):
            if self.path != "/api/save":
                return self.send(404, b"Not found", "text/plain")
            if self.headers.get("Origin") != f"http://{self.headers.get('Host')}":
                return self.send(403, b"Same-origin required", "text/plain")
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 1_000_000:
                    raise ValueError("invalid_body_size")
                body = json.loads(self.rfile.read(size))
                if body["packet_sha256"] != packet["packet_sha256"]:
                    raise ValueError("review_packet_changed")
                verify_images(packet)
                content = csv_bytes(packet, body["reviews"])
                checksum = hashlib.sha256(content).hexdigest()
                target = directory / "submissions" / f"review-{checksum}.csv"
                fixed(target, content)
                report = reliability(packet, body["reviews"])
                write_once(target.with_suffix(".report.json"), report)
                return self.send(
                    200, encode({"path": str(target.relative_to(ROOT)), "report": report}), "application/json"
                )
            except (KeyError, ValueError, TypeError) as error:
                return self.send(400, encode({"error": str(error)}), "application/json")

    with ThreadingHTTPServer(("127.0.0.1", port), Handler) as server:
        print(f"Review: http://127.0.0.1:{server.server_port}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
