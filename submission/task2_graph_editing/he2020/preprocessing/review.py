"""Image-free human audit with existing five-axis rules and bound evidence."""

import csv
import io
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from task2_graph_editing.he2020.preprocessing.data import checked_inputs
from common.io import digest, encode, fixed, read, sha256, write

from task2_graph_editing.magicbrush.review_interface import AXES, validate_reviews

FIELDS = (
    "pair_id",
    "split",
    "original_record_index",
    "evidence_sha256",
    "review_packet_sha256",
    "decision",
    *AXES,
    "reviewer",
    "notes",
)
ASSETS = Path(__file__).with_name("review_assets")


def make_packet(output):
    manifest, index, labels = checked_inputs(output)
    diagnostic = manifest["review_ids"]
    conditional = sorted(
        (i for i, r in labels.items() if r["status"] == "candidate" and r["split"] == "train"),
        key=lambda i: digest(["conditional-review", 42, i]),
    )[:20]
    cases = []
    for identifier in dict.fromkeys(diagnostic + conditional):
        label = labels[identifier]
        cases.append(
            {
                **index[identifier],
                "label": label,
                "graphs_ready": bool(label["stages"]["transaction_validity"]),
                "evidence_sha256": digest(label),
            }
        )
    packet = {
        "schema": "emnlp2020_graph_review_v1",
        "dataset_manifest_sha256": sha256(Path(output) / "manifest.json"),
        "edits_sha256": sha256(Path(output) / "edits.json"),
        "cases": cases,
        "diagnostic_review_ids": diagnostic,
        "conditional_review_ids": conditional,
        "axes": list(AXES),
        "export_fields": list(FIELDS),
        "selection": "20 train IDs before quality filtering plus up to 20 independently hash-selected train candidates; overlap reviewed once",
        "review_scope": "faithfulness to official graphs and instruction; real-image correctness not evaluated",
        "human_reviews_created": 0,
        "automatic_approval": False,
    }
    packet["packet_sha256"] = digest(packet)
    return packet


def validate(packet, reviews):
    # Failed examples are reviewable as incorrect/uncertain: original graphs
    # and failure evidence remain present. They can never become correct.
    validate_reviews({**packet, "cases": [{**r, "graphs_ready": True} for r in packet["cases"]]}, reviews)
    for row in packet["cases"]:
        if reviews.get(row["id"], {}).get("decision") == "correct" and row["label"]["status"] != "candidate":
            raise ValueError("failed_candidate_cannot_be_human_correct")


def csv_bytes(packet, reviews):
    validate(packet, reviews)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\r\n")
    writer.writeheader()
    for row in packet["cases"]:
        writer.writerow(
            {
                "pair_id": row["id"],
                "split": row["split"],
                "original_record_index": row["original_record_index"],
                "evidence_sha256": row["evidence_sha256"],
                "review_packet_sha256": packet["packet_sha256"],
                **{k: reviews.get(row["id"], {}).get(k, "") for k in ("decision", *AXES, "reviewer", "notes")},
            }
        )
    return stream.getvalue().encode("utf-8-sig")


def reliability(packet, reviews):
    validate(packet, reviews)

    def stats(ids):
        counts = Counter(reviews.get(i, {}).get("decision") or "unreviewed" for i in ids)
        determinate = counts["correct"] + counts["incorrect"]
        return {
            "size": len(ids),
            "decision_counts": dict(counts),
            "determinate": determinate,
            "correct_fraction_among_determinate": counts["correct"] / determinate if determinate else None,
            "correct_fraction_entire_cohort": counts["correct"] / len(ids)
            if ids and not counts["unreviewed"]
            else None,
            "complete": bool(ids) and not counts["unreviewed"],
        }

    diagnostic, conditional = (stats(packet[k]) for k in ("diagnostic_review_ids", "conditional_review_ids"))
    ready = (
        diagnostic["size"] >= 20
        and diagnostic["complete"]
        and conditional["determinate"] >= 20
        and conditional["correct_fraction_among_determinate"] >= 0.8
    )
    return {
        "ready": ready,
        "diagnostic": diagnostic,
        "candidate_conditional": conditional,
        "reviewed_pairs": sum(bool(r.get("decision")) for r in reviews.values()),
        "rejected_ids": sorted(i for i, r in reviews.items() if r.get("decision") in ("incorrect", "uncertain")),
        "minimum_determinate_candidate_reviews": 20,
        "minimum_correct_fraction": 0.8,
        "review_packet_sha256": packet["packet_sha256"],
        "automatically_approved": False,
        "image_accuracy": None,
        "inter_rater_agreement": None,
    }


def build(output, directory):
    directory = Path(directory)
    packet = make_packet(output)
    write(directory / "review-data.json", packet)
    write(directory / "packets" / f"{packet['packet_sha256']}.json", packet)
    fixed(directory / "review-template.csv", csv_bytes(packet, {}))
    for name in ("index.html", "viewer.js", "viewer.css"):
        content = (ASSETS / name).read_bytes()
        if name == "index.html":
            content = content.replace(
                b"__REVIEW_DATA__", encode(packet).replace(b"<", b"\\u003c").replace(b"&", b"\\u0026")
            )
        fixed(directory / name, content)
    return {
        "directory": str(directory),
        "cases": len(packet["cases"]),
        "packet_sha256": packet["packet_sha256"],
        "human_reviews_created": 0,
    }


def checked_packet(output, directory):
    packet = read(Path(directory) / "review-data.json")
    if packet != make_packet(output):
        raise ValueError("review_evidence_or_selection_changed")
    return packet


def audit(output, directory, csv_path=None):
    packet = checked_packet(output, directory)
    reviews = {}
    if csv_path is not None:
        with Path(csv_path).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != list(FIELDS):
                raise ValueError("unexpected_review_csv_schema")
            rows = list(reader)
        expected = {r["pair_id"]: r for r in csv.DictReader(io.StringIO(csv_bytes(packet, {}).decode("utf-8-sig")))}
        if len(rows) != len(expected) or {r["pair_id"] for r in rows} != set(expected):
            raise ValueError("duplicate_missing_or_unknown_review_ids")
        mutable = {"decision", *AXES, "reviewer", "notes"}
        for row in rows:
            if any(row[k] != expected[row["pair_id"]][k] for k in set(FIELDS) - mutable):
                raise ValueError("review_csv_evidence_mismatch")
            reviews[row["pair_id"]] = {k: row[k] for k in mutable}
    result = reliability(packet, reviews)
    result["review_csv_sha256"] = sha256(csv_path) if csv_path else None
    write(Path(directory) / "audits" / f"{digest(result)}.json", result)
    return result


def handler(output, directory):
    directory = Path(directory)
    packet = checked_packet(output, directory)

    class Handler(BaseHTTPRequestHandler):
        def send(self, status, content, kind):
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            names = {
                "/": ("index.html", "text/html; charset=utf-8"),
                "/viewer.js": ("viewer.js", "text/javascript; charset=utf-8"),
                "/viewer.css": ("viewer.css", "text/css"),
            }
            if self.path not in names:
                return self.send(404, b"Not found", "text/plain")
            name, kind = names[self.path]
            return self.send(200, (directory / name).read_bytes(), kind)

        def do_POST(self):
            if self.path != "/api/save" or self.headers.get("Origin") != f"http://{self.headers.get('Host')}":
                return self.send(403, b"Same origin required", "text/plain")
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 1_000_000:
                    raise ValueError("invalid_body_size")
                import json

                body = json.loads(self.rfile.read(size))
                if body["packet_sha256"] != checked_packet(output, directory)["packet_sha256"]:
                    raise ValueError("review_packet_changed")
                content = csv_bytes(packet, body["reviews"])
                report = reliability(packet, body["reviews"])
                path = directory / "submissions" / f"review-{digest(body['reviews'])}.csv"
                fixed(path, content)
                write(path.with_suffix(".report.json"), report)
                return self.send(200, encode({"path": str(path), "report": report}), "application/json")
            except (ValueError, KeyError, TypeError) as error:
                return self.send(400, encode({"error": str(error)}), "application/json")

    return Handler


def serve(output, directory, port=8775):
    with ThreadingHTTPServer(("127.0.0.1", port), handler(output, directory)) as server:
        print(f"Review: http://127.0.0.1:{server.server_port}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
