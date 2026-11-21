"""Load an explicit reviewed CSV into the existing viewer without changing evidence.

Adds a versioned, one-time browser CSV import to index.html. The original
index is archived, and later browser edits survive reloads of the same snapshot.
No source template, packet, mapping, graph, embedding or training is changed.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

from scripts.prepare_knowledge import _save_outputs
from src.knowledge.transe.core import sha256
from tools.prepare_gqa_wordnet_review import EXPORT_FIELDS, ROOT, canonical_hash, export_csv, json_bytes, under

BEGIN = "<!-- BEGIN CONFIRMED GQA REVIEW SNAPSHOT -->"
END = "<!-- END CONFIRMED GQA REVIEW SNAPSHOT -->"
SCRIPT = Path(__file__).with_name("gqa_review_snapshot.js")


def validate_review(packet, csv_path):
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != EXPORT_FIELDS:
            raise ValueError("Review CSV schema differs")
        rows = list(reader)
    expected = {str(row["class_id"]): row for row in packet["classes"]}
    reviews = {}
    for row in rows:
        key = row["class_id"]
        if key not in expected or key in reviews:
            raise ValueError("Unknown or duplicate class ID")
        item = expected[key]
        for field in ("class_id", "label", "status", "synset_id", "definition", "candidates"):
            if row[field] != item["original"][field]:
                raise ValueError(f"Original {field} differs: {key}")
        if (
            row["review_packet_sha256"] != packet["packet_sha256"]
            or row["train_image_ids"] != ";".join(e["image_id"] for e in item["examples"])
            or row["evidence_status"] != item["evidence_status"]
        ):
            raise ValueError("Review evidence/version differs")
        if row["decision"] not in ("accept", "defer"):
            raise ValueError("A confirmed snapshot requires an explicit decision for every label")
        reviews[key] = {field: row[field] for field in ("decision", "selected_synset", "reviewer", "notes")}
    if set(reviews) != set(expected):
        raise ValueError("Review CSV must cover every sampled class exactly once")
    export_csv(packet, reviews)  # Existing reviewer, candidate and image-evidence validation.
    return rows


def apply_snapshot(output, csv_path):
    output = under(output, ROOT / "reviews")
    packet_path, index_path = output / "review-data.json", output / "index.html"
    packet = json.loads(packet_path.read_text())
    if canonical_hash({k: v for k, v in packet.items() if k != "packet_sha256"}) != packet["packet_sha256"]:
        raise ValueError("Review packet content hash mismatch")
    rows = validate_review(packet, csv_path)
    content = index_path.read_text()
    pattern = re.escape(BEGIN) + r"\n.*?" + re.escape(END) + r"\n"
    if content.count(BEGIN) != content.count(END) or content.count(BEGIN) > 1:
        raise ValueError("Malformed snapshot block")
    baseline = re.sub(pattern, "", content, flags=re.DOTALL)
    verification = json.loads((output / "verification.json").read_text())
    baseline_sha = verification["files"][str(index_path.relative_to(ROOT))]
    if hashlib.sha256(baseline.encode()).hexdigest() != baseline_sha:
        raise ValueError("Viewer HTML differs from the verified baseline outside the snapshot block")
    checksum = sha256(csv_path)
    snapshot = {
        "schema_version": 1,
        "packet_sha256": packet["packet_sha256"],
        "csv_sha256": checksum,
        "csv_name": csv_path.name,
        "csv_text": csv_path.read_text(encoding="utf-8"),
        "application": "one_time_import_of_explicit_user_review_with_browser_backup",
    }
    encoded = json.dumps(snapshot, ensure_ascii=False).replace("<", "\\u003c").replace("&", "\\u0026")
    block = (
        f'{BEGIN}\n<script id="confirmed-review-snapshot" type="application/json">{encoded}</script>\n'
        f"<script>\n{SCRIPT.read_text()}</script>\n{END}\n"
    )
    if baseline.count("</body>") != 1:
        raise ValueError("Expected one HTML body")
    updated = baseline.replace("</body>", block + "</body>")
    _save_outputs(
        output,
        {
            "index-before-confirmed-review.html": baseline.encode(),
            f"review-snapshot-{checksum[:16]}.json": json_bytes(snapshot),
        },
        dry_run=False,
    )
    if content != updated:
        handle, temp_name = tempfile.mkstemp(prefix=".index-review-", dir=output)
        try:
            with os.fdopen(handle, "w") as stream:
                stream.write(updated)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, index_path)
        finally:
            Path(temp_name).unlink(missing_ok=True)
    counts = {decision: sum(row["decision"] == decision for row in rows) for decision in ("accept", "defer")}
    report = {
        "status": "PASS",
        "csv_path": str(csv_path),
        "csv_sha256": checksum,
        "packet_sha256": packet["packet_sha256"],
        "packet_file_sha256": sha256(packet_path),
        "decision_counts": counts,
        "baseline_index_sha256": baseline_sha,
        "updated_index_sha256": sha256(index_path),
        "browser_draft_backup_before_import": True,
        "same_snapshot_preserves_subsequent_browser_edits": True,
        "mapping_updated": False,
        "training_launched": False,
        "source_sha256": {str(p): sha256(p) for p in (Path(__file__), SCRIPT)},
    }
    _save_outputs(output, {f"viewer-update-{checksum[:16]}.json": json_bytes(report)}, dry_run=False)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "reviews/gqa-wordnet-human-v1")
    args = parser.parse_args(argv)
    report = apply_snapshot(args.output, args.csv)
    print(json.dumps({"status": report["status"], "decision_counts": report["decision_counts"]}))


if __name__ == "__main__":
    main()
