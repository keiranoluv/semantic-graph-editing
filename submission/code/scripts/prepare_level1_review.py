"""Prepare a human review worksheet without filling any human decisions."""

import argparse
import csv
import io
import json
from pathlib import Path

from scripts.prepare_knowledge import _save_outputs
from src.knowledge.sense_review import REVIEW_FIELDS, _sha256_json, _validate_mapping
from src.knowledge.transe.core import sha256


def prepare_rows(mapping, draft, deferred):
    original = _validate_mapping(mapping)
    proposed = _validate_mapping(draft)
    if len(original) != 151 or len(proposed) != 151:
        raise ValueError("Expected 151 VG150 rows including background")
    if draft.get("source_mapping_canonical_sha256") != _sha256_json(mapping):
        raise ValueError("Draft is not derived from the supplied original mapping")
    deferred_by_id = {}
    for item in deferred:
        try:
            class_id = int(item["class_id"])
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError("Invalid deferred class ID") from exc
        if class_id in deferred_by_id or not 1 <= class_id <= 150:
            raise ValueError("Duplicate/out-of-range deferred class ID")
        deferred_by_id[class_id] = item
    result = []
    for row, suggestion in zip(original[1:], proposed[1:], strict=True):
        if row["label"] != suggestion["label"]:
            raise ValueError("Draft/original class labels differ")
        candidates = {c["synset_id"]: c.get("definition", "") for c in row["candidates"]}
        class_id = row["class_id"]
        pending = deferred_by_id.pop(class_id, None)
        mapped = suggestion.get("synset_id") is not None
        if mapped and pending is not None or not mapped and pending is None:
            raise ValueError(f"Deferred CSV and selected draft mapping disagree at class {class_id}")
        if pending is not None and (
            pending.get("label") != row["label"] or suggestion["status"] not in {"ambiguous", "unmatched"}
        ):
            raise ValueError(f"Deferred CSV class label/status mismatch at class {class_id}")
        proposed_synset = suggestion.get("synset_id") if mapped else pending.get("suggested_synset")
        if not isinstance(proposed_synset, str) or proposed_synset not in candidates:
            raise ValueError("Suggested synset is outside original WordNet candidates")
        result.append(
            {
                "class_id": row["class_id"],
                "label": row["label"],
                "status": row["status"],
                "suggested_synset": proposed_synset,
                "selected_synset": "",
                "decision": "",
                "reviewer": "",
                "notes": "",
                "suggested_definition": candidates[proposed_synset],
                "all_candidates": json.dumps(candidates, ensure_ascii=False),
                "draft_mapping_state": "selected_provisional" if mapped else "deferred_unmapped",
                "deferred_reason": "" if mapped else pending.get("reason", ""),
            }
        )
    if deferred_by_id:
        raise ValueError("Deferred CSV contains classes absent from the mapping")
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mapping", type=Path, default=Path("data/knowledge/wordnet-vg150/wordnet_mapping.json"))
    p.add_argument(
        "--draft", type=Path, default=Path("data/knowledge/wordnet-ai-draft-v1/mapping/wordnet_mapping.json")
    )
    p.add_argument(
        "--deferred-csv", type=Path, default=Path("data/knowledge/wordnet-ai-draft-v1/mapping/deferred_review.csv")
    )
    p.add_argument("--output-dir", type=Path, default=Path("data/knowledge/level1-human-review-v2"))
    args = p.parse_args(argv)
    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    draft = json.loads(args.draft.read_text(encoding="utf-8"))
    with args.deferred_csv.open(encoding="utf-8-sig", newline="") as stream:
        deferred = list(csv.DictReader(stream))
    rows = prepare_rows(mapping, draft, deferred)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=[*REVIEW_FIELDS, "suggested_definition", "all_candidates", "draft_mapping_state", "deferred_reason"],
    )
    writer.writeheader()
    writer.writerows(rows)
    _save_outputs(
        args.output_dir,
        {
            "review.csv": stream.getvalue().encode("utf-8-sig"),
            "manifest.json": (
                json.dumps(
                    {
                        "mapping_sha256": sha256(args.mapping),
                        "draft_sha256": sha256(args.draft),
                        "deferred_csv_sha256": sha256(args.deferred_csv),
                        "review_status": "template_only_no_human_decisions",
                        "class_count": len(rows),
                        "suggestion_count": sum(bool(row["suggested_synset"]) for row in rows),
                        "deferred_count": sum(row["draft_mapping_state"] == "deferred_unmapped" for row in rows),
                    },
                    indent=2,
                )
                + "\n"
            ).encode("utf-8"),
        },
        dry_run=False,
    )


if __name__ == "__main__":
    main()
