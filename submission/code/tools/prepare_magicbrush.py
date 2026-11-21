"""CPU-only pinned MagicBrush train subset, image provenance and grouped splits."""

import argparse
import csv
import hashlib
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image

from scripts.prepare_knowledge import _save_outputs
from src.editing.splits import grouped_split
from src.knowledge.transe.core import sha256

ROOT = Path(__file__).resolve().parents[1]
REVISION = "6c254560879df7851233ea0fd2ae2a62764d0ef2"
REPO = "osunlp/MagicBrush"
RAW = ROOT / "data/raw/magicbrush-hf-6c254560"
DATA = ROOT / "data/processed/level2/magicbrush-v1"
REPORT = ROOT / "reports/level2/magicbrush-v1"


def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def fixed(path, content):
    _save_outputs(path.parent, {path.name: content}, dry_run=False)


def download_shard(entry, output):
    name = Path(entry["path"]).name
    if not entry["path"].startswith("data/train-") or Path(entry["path"]).parent.as_posix() != "data":
        raise ValueError("Only official train shards are allowed")
    target = output / name
    expected = entry["lfs"]["oid"]
    if target.exists():
        if target.stat().st_size != entry["size"] or sha256(target) != expected:
            raise ValueError(f"Existing shard checksum differs: {target}")
        print(f"[VERIFIED] {name}", flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".parquet.part")
    response = requests.get(
        f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{entry['path']}", stream=True, timeout=(30, 90)
    )
    response.raise_for_status()
    with response, partial.open("wb") as stream:
        for block in response.iter_content(1024 * 1024):
            stream.write(block)
    if partial.stat().st_size != entry["size"] or sha256(partial) != expected:
        raise ValueError(f"Downloaded shard size/SHA256 differs: {name}; partial preserved")
    os.link(partial, target)
    partial.unlink()
    print(f"[DOWNLOADED] {name} ({entry['size']} bytes)", flush=True)


def download(output=RAW, shard_count=4):
    metadata_path = output / "source.json"
    if metadata_path.exists():
        source = json.loads(metadata_path.read_text())
        if source["revision"] != REVISION or len(source["shards"]) != shard_count:
            raise ValueError("Use a fresh raw directory for a different source selection")
    else:
        response = requests.get(
            f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/data",
            params={"recursive": "false", "expand": "false"},
            timeout=30,
        )
        response.raise_for_status()
        shards = sorted((e for e in response.json() if e["path"].startswith("data/train-")), key=lambda e: e["path"])[
            :shard_count
        ]
        if len(shards) != shard_count or any("lfs" not in e for e in shards):
            raise ValueError("Incomplete pinned train shard metadata")
        card = requests.get(f"https://huggingface.co/datasets/{REPO}/raw/{REVISION}/README.md", timeout=30)
        card.raise_for_status()
        source = {
            "schema_version": 1,
            "repo": REPO,
            "revision": REVISION,
            "official_split": "train",
            "selection": f"first {shard_count} train shards by pinned filename, not a random sample of the whole dataset",
            "shards": shards,
            "license_declared_by_dataset_card": "CC-BY-4.0",
            "dataset_card_sha256": hashlib.sha256(card.content).hexdigest(),
            "official_dev_or_test_files_downloaded": False,
        }
        fixed(output / "README.upstream.md", card.content)
        fixed(metadata_path, encode(source))
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda entry: download_shard(entry, output), source["shards"]))
    return source


def choose_groups(rows, minimum=600, seed=42):
    groups = {}
    seen = set()
    for row in rows:
        key = (row["img_id"], row["turn_index"])
        if key in seen or not row["img_id"] or not row["instruction"].strip():
            raise ValueError("Duplicate ID/turn or empty source metadata")
        seen.add(key)
        groups.setdefault(row["img_id"], []).append(row)
    selected = []
    for group in sorted(groups, key=lambda g: (digest([seed, g]), g)):
        selected.extend(sorted(groups[group], key=lambda r: r["turn_index"]))
        if len(selected) >= minimum:
            break
    if len(selected) < minimum:
        raise ValueError(f"Only {len(selected)} candidate turns; requested at least {minimum}")
    return selected


def image_record(blob, output):
    if not blob:
        raise ValueError("Image bytes are missing")
    checksum = hashlib.sha256(blob).hexdigest()
    with Image.open(io.BytesIO(blob)) as image:
        image.load()
        width, height = image.size
        extension = {"PNG": "png", "JPEG": "jpg", "WEBP": "webp"}.get(image.format)
        if extension is None or min(width, height) <= 0:
            raise ValueError("Unsupported image encoding/dimensions")
        rgb = image.convert("RGB")
        pixel_hash = hashlib.sha256(f"{width}:{height}:RGB:".encode() + rgb.tobytes()).hexdigest()
    path = output / "images" / f"{checksum}.{extension}"
    fixed(path, blob)
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": checksum,
        "pixel_sha256": pixel_hash,
        "width": width,
        "height": height,
        "bytes": len(blob),
    }


def split_records(records, seed=42):
    """Union sessions sharing any source/target pixels before assigning splits."""
    parent = {r["img_id"]: r["img_id"] for r in records}

    def root(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    owners = {}
    for row in records:
        for role in ("source_image", "target_image"):
            checksum = row[role]["pixel_sha256"]
            if checksum in owners:
                a, b = sorted((root(row["img_id"]), root(owners[checksum])))
                parent[b] = a
            else:
                owners[checksum] = row["img_id"]
    for row in records:
        row["source_group_id"] = "coco:" + root(row["img_id"])
    splits = grouped_split(records, seed=seed, val_fraction=0.15, test_fraction=0.15)
    membership = {identifier: name for name, ids in splits.items() for identifier in ids}
    for row in records:
        row["split"] = membership[row["id"]]

    def pixels(split):
        return {
            r[role]["pixel_sha256"] for r in records if r["split"] == split for role in ("source_image", "target_image")
        }

    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if pixels(a) & pixels(b):
            raise AssertionError("Shared image pixels across project splits")
    return splits


def prepare(raw=RAW, output=DATA, report=REPORT, minimum=600, seed=42):
    import pyarrow.parquet as pq

    raw, output, report = raw.resolve(), output.resolve(), report.resolve()
    source = json.loads((raw / "source.json").read_text())
    metadata = []
    for entry in source["shards"]:
        path = raw / Path(entry["path"]).name
        if sha256(path) != entry["lfs"]["oid"]:
            raise ValueError("Raw shard hash differs")
        table = pq.read_table(path, columns=["img_id", "turn_index", "instruction"])
        for index, row in enumerate(table.to_pylist()):
            metadata.append({**row, "shard": path.name, "row_index": index})
    selected = choose_groups(metadata, minimum, seed)
    wanted = {(r["shard"], r["row_index"]): r for r in selected}
    records = []
    for entry in source["shards"]:
        path = raw / Path(entry["path"]).name
        offset = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8, columns=["source_img", "target_img"]):
            for local, images in enumerate(batch.to_pylist()):
                key = (path.name, offset + local)
                if key not in wanted:
                    continue
                row = wanted[key]
                records.append(
                    {
                        "id": f"magicbrush:{row['img_id']}:{row['turn_index']}",
                        "img_id": row["img_id"],
                        "turn_index": row["turn_index"],
                        "instruction": row["instruction"],
                        "official_split": "train",
                        "source_shard": row["shard"],
                        "source_row_index": row["row_index"],
                        "source_image": image_record(images["source_img"]["bytes"], output),
                        "target_image": image_record(images["target_img"]["bytes"], output),
                    }
                )
            offset += batch.num_rows
    records.sort(key=lambda r: r["id"])
    splits = split_records(records, seed)
    review_ids = sorted(splits["train"], key=lambda i: (digest(["magicbrush-review-v1", seed, i]), i))[:20]
    if len(review_ids) != 20:
        raise ValueError("Not enough train examples for fixed 20-pair review")
    manifest = {
        "schema_version": 1,
        "dataset": REPO,
        "revision": REVISION,
        "official_source": "train_only",
        "project_split_policy": "grouped 70/15/15 from selected official train; not official dev/test benchmark",
        "seed": seed,
        "records": records,
        "splits": splits,
        "review_ids": review_ids,
        "selection": source["selection"]
        + "; whole img_id groups within downloaded cohort in SHA256(seed,img_id) order",
        "session_completeness": "all turns present in downloaded shards; not asserted complete outside the shard cohort",
        "source_metadata_sha256": sha256(raw / "source.json"),
        "graph_labels_generated": False,
        "minimum_valid_graph_edit_pairs": 300,
        "graph_edit_requirement_complete": False,
    }
    fixed(output / "manifest.json", encode(manifest))
    summary = {
        "status": "PASS",
        "scope": "real_raw_edit_pairs_preparation_not_graph_labels",
        "pairs": len(records),
        "source_groups": len({r["source_group_id"] for r in records}),
        "split_counts": {k: len(v) for k, v in splits.items()},
        "review_pairs": 20,
        "unique_images": len({r[role]["sha256"] for r in records for role in ("source_image", "target_image")}),
        "pixel_overlap_between_project_splits": False,
        "official_dev_test_opened": False,
        "dataset_revision": REVISION,
        "manifest_sha256": sha256(output / "manifest.json"),
        "graph_labels_generated": False,
        "human_review_completed": False,
        "source_shards": [{"path": e["path"], "sha256": e["lfs"]["oid"], "bytes": e["size"]} for e in source["shards"]],
    }
    fixed(report / "preparation.json", encode(summary))
    fixed(
        report / "split-ids.json",
        encode(
            {
                "seed": seed,
                "dataset_revision": REVISION,
                "official_source": "train_only",
                "splits": splits,
                "source_groups": {r["id"]: r["source_group_id"] for r in records},
            }
        ),
    )
    stream = io.StringIO(newline="")
    fields = ("id", "img_id", "turn_index", "source_group_id", "split", "source_sha256", "target_sha256", "instruction")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(
        {
            **{k: r[k] for k in ("id", "img_id", "turn_index", "source_group_id", "split", "instruction")},
            "source_sha256": r["source_image"]["sha256"],
            "target_sha256": r["target_image"]["sha256"],
        }
        for r in records
    )
    fixed(report / "pair-manifest.csv", stream.getvalue().encode())
    fixed(
        report / "review-cohort.json",
        encode(
            {
                "seed": seed,
                "split": "train",
                "ids": review_ids,
                "selection": "lowest SHA256 over all train candidate IDs before graph generation",
            }
        ),
    )
    print(json.dumps(summary, indent=2), flush=True)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("download", "prepare", "all"), default="all")
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--output", type=Path, default=DATA)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--minimum-pairs", type=int, default=600)
    args = parser.parse_args(argv)
    if args.stage in ("all", "download"):
        download(args.raw)
    if args.stage in ("all", "prepare"):
        prepare(args.raw, args.output, args.report, args.minimum_pairs)


if __name__ == "__main__":
    main()
