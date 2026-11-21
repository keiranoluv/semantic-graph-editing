"""Rebuild the image-free VN100 Hugging Face table from the frozen evaluation."""

import json
from collections import Counter
from pathlib import Path

from tools.level3_submission_v1.bundle import sha
from tools.level3_submission_v1.demo import checked, open_bundle, recompute

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "submission/assets/datasets/vn100/data/dev.jsonl"


def main():
    root, manifest = open_bundle(ROOT)
    labels = json.loads(checked(root, manifest, manifest["labels"]).read_text())
    path = root / "configs/level3/qwen200-cohort-v1.json"
    assert sha(path) == labels["cohort_sha256"]
    tasks = json.loads(path.read_text())["tasks"]
    reviews = {r["task_id"]: r for r in labels["reviews"]}
    rows = []
    for task in tasks:
        tid = task["task_id"]
        case = recompute(root, manifest, tid, images=False)
        s = case["recomputed"]
        review = reviews[tid]
        source = manifest["sources"][task["image_id"]]
        assert case["image_id"] == source["image_id"] and case["instruction"] == task["instruction_en"]
        assert task["source_asset_url"] == source["upstream_url"]
        assert sha(root / source["original"]) == task["sha256"]
        edit, keep = review["edit_fulfillment"], review["preservation"]
        assert (edit, keep) == tuple(case["scores_and_labels"][k] for k in ("edit_fulfillment", "preservation"))
        rows.append(
            {
                "task_id": tid,
                "image_id": task["image_id"],
                "source_dataset": task["dataset"],
                "source_url": source["upstream_url"],
                "source_revision": source["revision"],
                "source_sha256": task["sha256"],
                "source_dataset_license": source["dataset_license"],
                "instruction": case["instruction"],
                "operation": task["operation"]["kind"],
                "edit_fulfillment": edit,
                "preservation": keep,
                "overall_pass": edit and keep,
                "defaulted_label_fields": labels["normalization"]["defaulted_fields"].get(tid, []),
                "edit_score": s["edit_score"],
                "preservation_score": s["preservation_score"],
                "preservation_evaluable": s["preservation_evaluable"],
                "graph_consistency": s["graph_consistency"],
                "reason_codes": s["reason_codes"],
            }
        )
    counts = Counter(r["image_id"] for r in rows)
    assert len({r["task_id"] for r in rows}) == 200 and len(counts) == 100 and set(counts.values()) == {2}
    assert sum(r["overall_pass"] for r in rows) == 175
    assert sum(r["graph_consistency"] is not None for r in rows) == 91
    assert sum(len(r["defaulted_label_fields"]) for r in rows) == 287
    OUTPUT.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows))


if __name__ == "__main__":
    main()
