"""Build/serve the fixed 20-pair MagicBrush review; never invent graph evidence."""

import argparse
import csv
import hashlib
import io
import json
import mimetypes
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from src.datasets.gqa_full import atomic_write
from src.editing.operations import EditGraph, Operation, apply_operations
from src.knowledge.transe.core import sha256
from tools.prepare_magicbrush import DATA, ROOT, digest, encode, fixed

TEMPLATES = Path(__file__).with_name("magicbrush_review_assets")
AXES = ("source_graph", "target_graph", "alignment", "operations", "instruction_consistency")
FIELDS = (
    "pair_id",
    "img_id",
    "turn_index",
    "split",
    "source_sha256",
    "target_sha256",
    "evidence_sha256",
    "decision",
    *AXES,
    "reviewer",
    "notes",
    "review_packet_sha256",
)
OUTPUT = ROOT / "reviews/magicbrush-human-v1"


def make_packet(manifest_path, edits_path):
    manifest = json.loads(manifest_path.read_text())
    edits = json.loads(edits_path.read_text())
    if edits["dataset_manifest_sha256"] != sha256(manifest_path):
        raise ValueError("Graph edits refer to another dataset manifest")
    index = {r["id"]: r for r in manifest["records"]}
    labels = {r["id"]: r for r in edits["records"]}
    if len(labels) != len(edits["records"]) or set(labels) - index.keys():
        raise ValueError("Repeated or unknown graph-edit pair IDs")
    selected = manifest["review_ids"]
    if len(selected) < 20 or len(set(selected)) != len(selected):
        raise ValueError("Review requires at least 20 distinct frozen IDs")
    cases = []
    for identifier in selected:
        row = index[identifier]
        if row["split"] != "train" or identifier not in manifest["splits"]["train"]:
            raise ValueError("Human review must use the preselected project train sample only")
        for role in ("source_image", "target_image"):
            if sha256(ROOT / row[role]["path"]) != row[role]["sha256"]:
                raise ValueError("Review image changed")
        label = labels.get(identifier)
        graph_ready = bool(label and "source_graph" in label and "target_graph" in label)
        if graph_ready:
            source = EditGraph.from_dict(label["source_graph"])
            target = EditGraph.from_dict(label["target_graph"])
            operations = tuple(Operation.from_dict(o) for o in label["operations"])
            if apply_operations(source, operations) != target:
                raise ValueError("Graph-edit evidence does not reconstruct the target graph")
        evidence = {
            "source_sha256": row["source_image"]["sha256"],
            "target_sha256": row["target_image"]["sha256"],
            "label": label,
        }
        cases.append({**row, "label": label, "graphs_ready": graph_ready, "evidence_sha256": digest(evidence)})
    packet = {
        "schema_version": 1,
        "dataset_manifest_sha256": sha256(manifest_path),
        "edits_file_sha256": sha256(edits_path),
        "cases": cases,
        "export_fields": FIELDS,
        "axes": AXES,
        "selection": "fixed 20 train IDs selected before any graph labels; no quality filtering",
        "graph_origin": "frozen_image_only_VLM_pseudo_annotations_not_GT",
        "human_reviews_created": 0,
    }
    packet["packet_sha256"] = digest(packet)
    return packet


def build(manifest_path, edits_path, output):
    output = output.resolve()
    if not output.is_relative_to(ROOT / "reviews"):
        raise ValueError("Review output must be under reviews/")
    packet = make_packet(manifest_path, edits_path)
    fixed(output / "packets" / f"{packet['packet_sha256']}.json", encode(packet))
    data = json.dumps(packet, ensure_ascii=False).replace("<", "\\u003c").replace("&", "\\u0026")
    page = (TEMPLATES / "index.html").read_text().replace("__REVIEW_DATA__", data)
    # These are derived viewer files. Every evidence packet is archived above;
    # submitted human CSVs are never changed when graphs become available.
    atomic_write(output / "review-data.json", packet)
    (output / "index.html").write_text(page)
    for name in ("viewer.js", "viewer.css"):
        (output / name).write_bytes((TEMPLATES / name).read_bytes())
    print(
        f"[READY] {len(packet['cases'])} real pairs; {sum(r['graphs_ready'] for r in packet['cases'])} with graph evidence: {output/'index.html'}"
    )
    return packet


def validate_reviews(packet, reviews):
    known = {r["id"]: r for r in packet["cases"]}
    if not isinstance(reviews, dict) or set(reviews) - known.keys():
        raise ValueError("Unknown review pair IDs")
    for identifier, item in reviews.items():
        if not isinstance(item, dict) or any(
            not isinstance(item.get(k, ""), str) for k in ("decision", *AXES, "reviewer", "notes")
        ):
            raise ValueError("Review fields must be strings")
        verdict = item.get("decision", "")
        if verdict not in ("", "correct", "incorrect", "uncertain"):
            raise ValueError("Invalid review verdict")
        if any(item.get(k, "") not in ("", "correct", "incorrect", "uncertain") for k in AXES):
            raise ValueError("Invalid review axis")
        if verdict:
            if not known[identifier]["graphs_ready"]:
                raise ValueError("Cannot review graphs that have not been generated")
            if not item.get("reviewer", "").strip() or any(not item.get(k) for k in AXES):
                raise ValueError("Verdict requires a reviewer and all five quality axes")
            if verdict == "correct" and any(item[k] != "correct" for k in AXES):
                raise ValueError("Overall correct contradicts axis ratings")
            if verdict == "incorrect" and not any(item[k] == "incorrect" for k in AXES):
                raise ValueError("Overall incorrect requires at least one incorrect axis")


def export_reviews(packet, reviews):
    validate_reviews(packet, reviews)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\r\n")
    writer.writeheader()
    for row in packet["cases"]:
        item = reviews.get(row["id"], {})
        writer.writerow(
            {
                "pair_id": row["id"],
                "img_id": row["img_id"],
                "turn_index": row["turn_index"],
                "split": row["split"],
                "source_sha256": row["source_image"]["sha256"],
                "target_sha256": row["target_image"]["sha256"],
                "evidence_sha256": row["evidence_sha256"],
                **{k: item.get(k, "") for k in ("decision", *AXES, "reviewer", "notes")},
                "review_packet_sha256": packet["packet_sha256"],
            }
        )
    return stream.getvalue().encode("utf-8-sig")


def reliability(packet, reviews):
    validate_reviews(packet, reviews)
    counts = Counter(
        item.get("decision", "") or "unreviewed"
        for identifier in (r["id"] for r in packet["cases"])
        for item in [reviews.get(identifier, {})]
    )
    denominator = counts["correct"] + counts["incorrect"]
    return {
        "schema_version": 1,
        "sample_size": len(packet["cases"]),
        "decision_counts": dict(counts),
        "reviewed_pairs": len(packet["cases"]) - counts["unreviewed"],
        "correct_fraction_among_determinate": counts["correct"] / denominator if denominator else None,
        "interpretation": "Human assessment of preselected train pseudo labels; not benchmark model performance",
        "review_packet_sha256": packet["packet_sha256"],
        "automatic_approval": False,
    }


def handler(output, packet):
    images = {(ROOT / r[role]["path"]).resolve() for r in packet["cases"] for role in ("source_image", "target_image")}

    class Handler(BaseHTTPRequestHandler):
        def send(self, code, content, kind):
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            name = unquote(urlsplit(self.path).path)
            if name in ("/", "/index.html", "/viewer.js", "/viewer.css", "/review-data.json"):
                path = output / (name.lstrip("/") or "index.html")
            else:
                path = (ROOT / name.lstrip("/")).resolve()
                if path not in images:
                    return self.send(404, b"Not found", "text/plain")
            if not path.is_file():
                return self.send(404, b"Not found", "text/plain")
            self.send(200, path.read_bytes(), mimetypes.guess_type(path.name)[0] or "application/octet-stream")

        def do_POST(self):
            if self.path != "/api/save":
                return self.send(404, b"Not found", "text/plain")
            if self.headers.get("Origin") != f"http://{self.headers.get('Host')}":
                return self.send(403, b"Same-origin required", "text/plain")
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 1_000_000:
                    raise ValueError("Invalid body size")
                body = json.loads(self.rfile.read(size))
                if body["packet_sha256"] != packet["packet_sha256"]:
                    raise ValueError("Review evidence version differs")
                content = export_reviews(packet, body["reviews"])
                checksum = hashlib.sha256(content).hexdigest()
                csv_name = f"review-{checksum[:16]}.csv"
                report = reliability(packet, body["reviews"])
                fixed(output / "submissions" / csv_name, content)
                fixed(output / "submissions" / f"review-{checksum[:16]}.report.json", encode(report))
                self.send(
                    200,
                    encode({"path": str((output / "submissions" / csv_name).relative_to(ROOT)), "report": report}),
                    "application/json",
                )
            except (KeyError, ValueError, TypeError) as error:
                self.send(400, encode({"error": str(error)}), "application/json")

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("build", "serve"), required=True)
    parser.add_argument("--manifest", type=Path, default=DATA / "manifest.json")
    parser.add_argument("--edits", type=Path, default=DATA / "edits-pending-v1.json")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args(argv)
    if args.stage == "build":
        build(args.manifest, args.edits, args.output)
    else:
        output = args.output.resolve()
        packet = json.loads((output / "review-data.json").read_text())
        if digest({k: v for k, v in packet.items() if k != "packet_sha256"}) != packet["packet_sha256"]:
            raise ValueError("Review packet hash differs")
        with ThreadingHTTPServer(("127.0.0.1", args.port), handler(output, packet)) as server:
            print(f"MagicBrush review: http://127.0.0.1:{server.server_port}/", flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    main()
