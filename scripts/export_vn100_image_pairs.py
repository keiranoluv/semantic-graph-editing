"""Build the image-pairs/dev Hub configuration from verified VN100 artifacts."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from tools.level3_submission_v1.demo import checked, open_bundle

ROOT = Path(__file__).resolve().parents[1]
HUB = ROOT / "submission/code/datasets/vn100"
OUTPUT = ROOT / "runs/level3/vn100-hf/image-pairs"
SHARD_SIZE = 25


def main():
    root, manifest = open_bundle(ROOT)
    credits = {row["image_id"]: row for row in map(json.loads, (HUB / "source-credits.jsonl").read_text().splitlines())}
    cases = {case["task_id"]: case for case in manifest["cases"]}
    metadata = list(map(json.loads, (HUB / "data/dev.jsonl").read_text().splitlines()))
    assert len(cases) == len(metadata) == 200 and len(credits) == 100
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(metadata), SHARD_SIZE):
        batch = []
        for row in metadata[start : start + SHARD_SIZE]:
            case = cases[row["task_id"]]
            original = manifest["sources"][row["image_id"]]["original"]
            source = checked(root, manifest, original)
            target = checked(root, manifest, case["target"])
            receipt = json.loads((target.parent / "result.json").read_text())
            target_hash = manifest["files"][case["target"]]["sha256"]
            credit = credits[row["image_id"]]
            assert receipt["status"] == "ok" and receipt["task_id"] == row["task_id"]
            assert row["image_id"] == receipt["image_id"] and row["instruction"] == receipt["prompt"]
            assert receipt["target"]["sha256"] == target_hash
            assert credit["source_sha256"] == row["source_sha256"] == manifest["files"][original]["sha256"]
            batch.append(
                {
                    "task_id": row["task_id"],
                    "instruction": row["instruction"],
                    "source_image": {"bytes": source.read_bytes(), "path": row["image_id"] + source.suffix},
                    "edited_image": {"bytes": target.read_bytes(), "path": row["task_id"] + target.suffix},
                    "edited_sha256": target_hash,
                    **credit,
                }
            )
        table = pa.Table.from_pylist(batch)
        features = {
            key: {"_type": "Image"}
            if key in ("source_image", "edited_image")
            else {"dtype": "string", "_type": "Value"}
            for key in table.column_names
        }
        table = table.replace_schema_metadata({b"huggingface": json.dumps({"info": {"features": features}}).encode()})
        pq.write_table(table, OUTPUT / f"dev-{start // SHARD_SIZE:02d}.parquet", row_group_size=4)


if __name__ == "__main__":
    main()
