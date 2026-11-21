"""Compile an explicitly supplied VG150 WordNet selection into reviewed overrides.

Selections are ordered by class ID 1..150. The output keeps the original
worksheet intact and records any extra WordNet candidate with provenance.
"""

import argparse
import copy
import csv
import hashlib
import io
import json
from pathlib import Path

from nltk.corpus import wordnet

from scripts.prepare_knowledge import _save_outputs
from src.knowledge.sense_review import _sha256_json, _validate_mapping, compile_reviews


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def read_selections(path):
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    if len(lines) != 151 or lines[0] != "selected_synset" or any(not value.strip() for value in lines[1:]):
        raise ValueError("Selection file must have a header and exactly 150 nonblank synset rows")
    return [value.strip() for value in lines[1:]]


def extend_mapping(mapping, additions, selections, *, reviewer, review_date):
    rows = _validate_mapping(mapping)
    if len(rows) != 151 or len(selections) != 150:
        raise ValueError("Expected 150 VG150 object classes")
    if mapping.get("source_version") != wordnet.get_version():
        raise ValueError("WordNet version differs from the source mapping")
    if additions.get("version") != 1 or additions.get("wordnet_version") != mapping["source_version"]:
        raise ValueError("Candidate additions need matching WordNet version and schema")
    extended = copy.deepcopy(mapping)
    additions_by_id = {}
    for item in additions.get("additions", []):
        class_id = item.get("class_id")
        if type(class_id) is not int or not 1 <= class_id <= 150 or class_id in additions_by_id:
            raise ValueError("Invalid or duplicated candidate addition class ID")
        if item.get("label") != rows[class_id]["label"]:
            raise ValueError(f"Candidate addition label mismatch at class {class_id}")
        synset_id = item.get("synset_id")
        if not isinstance(synset_id, str) or not item.get("reason"):
            raise ValueError(f"Candidate addition needs synset and reason at class {class_id}")
        synset = wordnet.synset(synset_id)
        if synset.pos() != "n" or synset.name() != synset_id:
            raise ValueError(f"Candidate addition is not a canonical noun synset: {synset_id}")
        if synset_id in {candidate["synset_id"] for candidate in rows[class_id]["candidates"]}:
            raise ValueError(f"Candidate addition is already present at class {class_id}")
        additions_by_id[class_id] = item
        extended["rows"][class_id]["candidates"].append({"synset_id": synset_id, "definition": synset.definition()})
        extended["rows"][class_id]["candidates"].sort(key=lambda candidate: candidate["synset_id"])
    used_additions = set()
    for class_id, choice in enumerate(selections, start=1):
        synset = wordnet.synset(choice)
        if synset.pos() != "n" or synset.name() != choice:
            raise ValueError(f"Class {class_id}: selection is not a canonical noun synset: {choice}")
        original_candidates = {candidate["synset_id"] for candidate in rows[class_id]["candidates"]}
        if choice not in original_candidates:
            addition = additions_by_id.get(class_id)
            if addition is None or addition["synset_id"] != choice:
                raise ValueError(f"Class {class_id}: selection outside source candidates without explicit addition")
            used_additions.add(class_id)
    if used_additions != set(additions_by_id):
        raise ValueError("Every candidate addition must be used by a reviewed selection")
    extended["review_candidate_additions"] = {
        "source_mapping_canonical_sha256": _sha256_json(mapping),
        "reviewer": reviewer,
        "review_date": review_date,
        "items": [additions_by_id[class_id] for class_id in sorted(additions_by_id)],
    }
    return extended


def review_csv(template_bytes, mapping, selections, notes_by_label, *, reviewer, review_date):
    reader = csv.DictReader(io.StringIO(template_bytes.decode("utf-8-sig"), newline=""))
    fieldnames = reader.fieldnames
    rows = list(reader)
    if not fieldnames or len(rows) != 150:
        raise ValueError("Review template must have exactly 150 rows")
    names = {row["label"] for row in rows}
    if any(label not in names or not isinstance(note, str) or not note for label, note in notes_by_label.items()):
        raise ValueError("Invalid per-class note")
    for class_id, (row, choice) in enumerate(zip(rows, selections, strict=True), start=1):
        expected = mapping["rows"][class_id]
        if (row["class_id"], row["label"], row["status"]) != (str(class_id), expected["label"], expected["status"]):
            raise ValueError(f"Review template is not aligned at class {class_id}")
        if any(row[field].strip() for field in ("selected_synset", "decision", "reviewer", "notes")):
            raise ValueError("Review template already contains decisions; use a fresh template")
        row["selected_synset"] = choice
        row["decision"] = "accept"
        row["reviewer"] = reviewer
        provenance = (
            f"{reviewer} supplied this selection as verified on {review_date}; "
            "the pasted column did not include a separate per-class rationale."
        )
        row["notes"] = provenance + (
            " Codex QA: " + notes_by_label[row["label"]] if row["label"] in notes_by_label else ""
        )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8-sig")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection-file",
        type=Path,
        default=Path("configs/knowledge/level1_verified_synsets_toan_2026-10-25.txt"),
    )
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--review-date", required=True)
    parser.add_argument("--template", type=Path, default=Path("data/knowledge/level1-human-review-v2/review.csv"))
    parser.add_argument("--mapping", type=Path, default=Path("data/knowledge/wordnet-vg150/wordnet_mapping.json"))
    parser.add_argument("--base-overrides", type=Path, default=Path("configs/knowledge/wordnet_overrides.json"))
    parser.add_argument(
        "--candidate-additions",
        type=Path,
        default=Path("configs/knowledge/level1_review_candidate_additions.json"),
    )
    parser.add_argument(
        "--notes-json",
        type=Path,
        default=Path("configs/knowledge/level1_review_notes_toan_2026-10-25.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/knowledge/level1-human-review-v2/compiled-toan-v1")
    )
    parser.add_argument(
        "--override-output",
        type=Path,
        default=Path("configs/knowledge/wordnet_overrides.reviewed-toan-v1.json"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.reviewer.strip() or not args.review_date.strip():
        parser.error("Reviewer and review date cannot be blank")
    wordnet.ensure_loaded()
    selections = read_selections(args.selection_file)
    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    additions = json.loads(args.candidate_additions.read_text(encoding="utf-8"))
    notes = json.loads(args.notes_json.read_text(encoding="utf-8"))
    base_overrides = json.loads(args.base_overrides.read_text(encoding="utf-8"))
    extended = extend_mapping(mapping, additions, selections, reviewer=args.reviewer, review_date=args.review_date)
    reviewed_csv = review_csv(
        args.template.read_bytes(), extended, selections, notes, reviewer=args.reviewer, review_date=args.review_date
    )
    overrides, report = compile_reviews(extended, reviewed_csv, base_overrides)
    if report["accepted_decisions"] != 150 or len(overrides["mappings"]) != 150:
        raise ValueError("Expected 150 explicit accepted decisions and overrides")
    report.update(
        reviewer=args.reviewer,
        review_date=args.review_date,
        selection_file_sha256=hashlib.sha256(args.selection_file.read_bytes()).hexdigest(),
        candidate_addition_count=len(additions["additions"]),
        note="Selections were supplied by reviewer in one ordered column; per-class rationale was not supplied for most rows.",
    )
    _save_outputs(
        args.output_dir,
        {
            "wordnet_mapping.review_candidates.json": json_bytes(extended),
            "review_human_toan.csv": reviewed_csv,
            "compile_report.json": json_bytes(report),
        },
        dry_run=args.dry_run,
    )
    _save_outputs(
        args.override_output.parent,
        {args.override_output.name: json_bytes(overrides)},
        dry_run=args.dry_run,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
