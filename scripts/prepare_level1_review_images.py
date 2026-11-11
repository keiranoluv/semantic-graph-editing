"""Copy downloaded review JPEGs to filenames that show their VG150 class."""

import argparse
import csv
import hashlib
import html
import json
import os
import re
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

from scripts.download_level1_review_images import ARCHIVES, FIELDS, archive_member, download_archive


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_label(label):
    name = re.sub(r"[^a-z0-9]+", "_", label.casefold()).strip("_")
    if not name:
        raise ValueError(f"Cannot make a filename from label {label!r}")
    return name


def review_filename(row):
    class_id, image_id = int(row["class_id"]), int(row["image_id"])
    return f"{class_id:03d}_{safe_label(row['label'])}_{image_id}.jpg"


def merge_review_rows(rows, extras):
    merged = {}
    for row in [*rows, *extras]:
        key = (int(row["class_id"]), int(row["image_id"]))
        if key in merged:
            previous = merged[key]
            if any(row[field] != previous[field] for field in ("label", "box_xyxy", "sha256")):
                raise ValueError(f"Conflicting review rows for class/image {key}")
        else:
            merged[key] = row
    return list(merged.values())


def add_examples(requests, rows, extra_rows, examples_path, extra_path, source_dir):
    with examples_path.open(encoding="utf-8-sig", newline="") as stream:
        examples = {(int(row["class_id"]), int(row["image_id"])): row for row in csv.DictReader(stream)}
    known = {(int(row["class_id"]), int(row["image_id"])) for row in [*rows, *extra_rows]}
    added = 0
    for request in requests:
        try:
            class_id, image_id = (int(value) for value in request.split(":"))
        except ValueError as exc:
            raise ValueError(f"Expected --add-example CLASS_ID:IMAGE_ID, got {request!r}") from exc
        key = (class_id, image_id)
        if key in known:
            continue
        if key not in examples:
            raise ValueError(f"Image {image_id} is not a sampled train example of class {class_id}")
        example = examples[key]
        archive_dir, _ = archive_member(example)
        source = source_dir / f"{image_id}.jpg"
        if source.is_file():
            data = source.read_bytes()
            if not data.startswith(b"\xff\xd8") or not data.endswith(b"\xff\xd9"):
                raise ValueError(f"Existing source is not a complete JPEG: {source}")
            entry = {
                "class_id": str(class_id),
                "label": example["label"],
                "image_id": str(image_id),
                "object_index": example["object_index"],
                "box_xyxy": example["box_xyxy"],
                "original_url": example["url"],
                "archive": ARCHIVES[archive_dir],
                "local_path": str(source),
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        else:
            entry = download_archive(archive_dir, [example], source_dir)[0]
        extra_rows.append(entry)
        known.add(key)
        added += 1
    if added:
        extra_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8-sig", newline="", dir=extra_path.parent, prefix=".review_extra_", delete=False
        ) as tmp:
            temporary = Path(tmp.name)
            writer = csv.DictWriter(tmp, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(sorted(extra_rows, key=lambda row: (int(row["class_id"]), int(row["image_id"]))))
        os.replace(temporary, extra_path)
    return added


def prepare_images(rows, output_dir):
    grouped = defaultdict(list)
    for row in rows:
        class_id = int(row["class_id"])
        if not 1 <= class_id <= 150:
            raise ValueError(f"Invalid class ID: {class_id}")
        grouped[class_id].append(row)
    if sorted(grouped) != list(range(1, 151)):
        raise ValueError("Review manifest must cover all 150 VG150 classes")

    planned = []
    for class_id in range(1, 151):
        labels = {row["label"] for row in grouped[class_id]}
        if len(labels) != 1:
            raise ValueError(f"Conflicting labels for class {class_id}")
        for index, row in enumerate(grouped[class_id], start=1):
            source = Path(row["local_path"])
            image_id = int(row["image_id"])
            if image_id <= 0 or source.name != f"{image_id}.jpg":
                raise ValueError(f"Invalid source image ID/path for class {class_id}")
            prefix = f"{class_id:03d}_{safe_label(row['label'])}"
            target = output_dir / review_filename(row)
            legacy = output_dir / f"{prefix}_{index:02d}.jpg"
            if not source.is_file() or sha256(source) != row["sha256"]:
                raise ValueError(f"Missing or changed source image: {source}")
            if target.exists() and sha256(target) != row["sha256"]:
                raise ValueError(f"Existing review image differs; refusing to overwrite: {target}")
            if legacy.exists() and sha256(legacy) != row["sha256"]:
                raise ValueError(f"Legacy review image differs; refusing to replace: {legacy}")
            planned.append((source, target, legacy))

    output_dir.mkdir(parents=True, exist_ok=True)
    copied = renamed = 0
    for source, target, legacy in planned:
        if target.exists():
            legacy.unlink(missing_ok=True)
            continue
        if legacy.exists():
            os.replace(legacy, target)
            renamed += 1
            continue
        with tempfile.NamedTemporaryFile(dir=output_dir, prefix=".review_", suffix=".part", delete=False) as tmp:
            temporary = Path(tmp.name)
        try:
            shutil.copyfile(source, temporary)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        copied += 1
    return len(planned), copied, renamed


def write_gallery(rows, output_dir, metadata_path):
    """Make a local HTML gallery with the target object box drawn over each JPEG."""
    metadata = {int(item["image_id"]): item for item in json.loads(metadata_path.read_text(encoding="utf-8"))}
    cards = []
    for row in sorted(rows, key=lambda item: (int(item["class_id"]), int(item["image_id"]))):
        image_id = int(row["image_id"])
        image = metadata[image_id]
        width, height = int(image["width"]), int(image["height"])
        x1, y1, x2, y2 = (float(value) for value in json.loads(row["box_xyxy"]))
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"Invalid box for image {image_id}")
        filename = html.escape(review_filename(row), quote=True)
        label = html.escape(row["label"])
        search = html.escape(f"{row['class_id']} {row['label']} {image_id}".casefold(), quote=True)
        cards.append(
            f'<article data-search="{search}">'
            f"<h2>{int(row['class_id']):03d} · {label}</h2>"
            f'<a href="{filename}" title="Open original JPEG">'
            f'<div class="photo"><img src="{filename}" loading="lazy" alt="{label}, image {image_id}">'
            f'<svg viewBox="0 0 {width} {height}" preserveAspectRatio="none" aria-hidden="true">'
            f'<rect x="{x1:.2f}" y="{y1:.2f}" width="{x2 - x1:.2f}" height="{y2 - y1:.2f}"/>'
            "</svg></div></a>"
            f"<p>image_id: {image_id} · box: [{x1:.0f}, {y1:.0f}]–[{x2:.0f}, {y2:.0f}]</p>"
            "</article>"
        )
    page = (
        '<!doctype html><html lang="vi"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>VG150 review images</title>"
        "<style>body{font:16px system-ui,sans-serif;margin:24px;background:#f5f5f5;color:#222}"
        "header{position:sticky;top:0;background:#f5f5f5;padding:8px 0;z-index:1}"
        "input{font:inherit;padding:10px;width:min(95%,400px)}"
        "main{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:16px}"
        "article{background:white;padding:14px;border-radius:10px;box-shadow:0 1px 5px #ccc}"
        "article[hidden]{display:none}h2{margin:0 0 10px;font-size:19px}p{margin:8px 0 0}"
        ".photo{position:relative}.photo img{display:block;width:100%;height:auto}"
        ".photo svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}"
        ".photo rect{fill:#ff304033;stroke:#ff3040;stroke-width:3;vector-effect:non-scaling-stroke}"
        "</style><header><h1>VG150 · ảnh review có hộp bao</h1>"
        "<p>Khung đỏ là vật thể có nhãn; ảnh có thể chứa vật thể khác cùng loại. "
        "Một ảnh chưa đủ để xác nhận nghĩa WordNet cho cả class.</p>"
        '<label>Lọc theo class ID, nhãn hoặc image ID: <input id="filter" type="search" autofocus></label>'
        "</header><main>" + "".join(cards) + "</main><script>const input=document.querySelector('#filter');"
        "input.addEventListener('input',()=>{const q=input.value.trim().toLowerCase();"
        "document.querySelectorAll('article').forEach(card=>card.hidden=!card.dataset.search.includes(q));});"
        "</script></html>"
    )
    target = output_dir / "index.html"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output_dir, prefix=".gallery_", delete=False
    ) as tmp:
        temporary = Path(tmp.name)
        tmp.write(page)
    os.replace(temporary, target)
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/knowledge/level1-human-review-v2/review_images.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/vg150/images_review"))
    parser.add_argument("--metadata", type=Path, default=Path("data/raw/vg150/image_data.json"))
    parser.add_argument(
        "--extra-manifest",
        type=Path,
        default=Path("data/knowledge/level1-human-review-v2/review_images_extra.csv"),
    )
    parser.add_argument(
        "--examples-csv",
        type=Path,
        default=Path("data/knowledge/level1-human-review-v2/train_examples_with_boxes.csv"),
    )
    parser.add_argument("--source-dir", type=Path, default=Path("data/raw/vg150/images"))
    parser.add_argument(
        "--add-example",
        action="append",
        default=[],
        metavar="CLASS_ID:IMAGE_ID",
        help="Add another sampled train image for this class to the review folder and gallery",
    )
    args = parser.parse_args(argv)
    with args.manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if args.extra_manifest.exists():
        with args.extra_manifest.open(encoding="utf-8-sig", newline="") as stream:
            extras = list(csv.DictReader(stream))
    else:
        extras = []
    added = add_examples(args.add_example, rows, extras, args.examples_csv, args.extra_manifest, args.source_dir)
    all_rows = merge_review_rows(rows, extras)
    total, copied, renamed = prepare_images(all_rows, args.output_dir)
    gallery = write_gallery(all_rows, args.output_dir, args.metadata)
    print(
        f"[OK] {total} review images in {args.output_dir}; "
        f"{copied} copied, {renamed} renamed, {added} new examples; gallery: {gallery}"
    )


if __name__ == "__main__":
    main()
