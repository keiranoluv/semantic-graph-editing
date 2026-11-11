"""Download sampled Visual Genome train JPEGs for VG150 class review.

The original Stanford image host may be unreachable. This script reads only
the requested JPEG members from a public mirror of the two original ZIPs via
HTTP byte ranges; it never downloads either complete multi-GB archive.
"""

import argparse
import csv
import hashlib
import io
import os
import re
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

MIRROR = "https://huggingface.co/datasets/pppop7/VisualGenome_VG_100K_1_and_2/resolve/main"
ARCHIVES = {"VG_100K": "images.zip", "VG_100K_2": "images2.zip"}
FIELDS = (
    "class_id",
    "label",
    "image_id",
    "object_index",
    "box_xyxy",
    "original_url",
    "archive",
    "local_path",
    "bytes",
    "sha256",
)


class HTTPRangeFile(io.RawIOBase):
    """Seekable, read-only view of a remote ZIP, backed by HTTP Range."""

    def __init__(self, url):
        with urlopen(Request(url, method="HEAD"), timeout=30) as response:
            self.url = response.url
            self.size = int(response.headers["Content-Length"])
        self.position = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=os.SEEK_SET):
        if whence == os.SEEK_SET:
            position = offset
        elif whence == os.SEEK_CUR:
            position = self.position + offset
        elif whence == os.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError("Invalid seek mode")
        if position < 0:
            raise ValueError("Cannot seek before start of archive")
        self.position = position
        return position

    def read(self, size=-1):
        if size < 0:
            size = self.size - self.position
        size = min(size, self.size - self.position)
        if size <= 0:
            return b""
        start, end = self.position, self.position + size - 1
        expected_range = f"bytes {start}-{end}/{self.size}"
        for attempt in range(3):
            try:
                with urlopen(Request(self.url, headers={"Range": f"bytes={start}-{end}"}), timeout=60) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != expected_range:
                        raise ValueError("Archive server did not honor the requested byte range")
                    data = response.read(size + 1)
                if len(data) != size:
                    raise OSError(f"Short archive read: expected {size}, received {len(data)}")
                self.position += size
                return data
            except (HTTPError, URLError, TimeoutError, OSError):
                if attempt == 2:
                    raise
                time.sleep(2**attempt)
        raise AssertionError("Unreachable")


def choose_distinct_examples(rows):
    """Find one unique image per class among the five train examples each."""
    by_class = defaultdict(list)
    for row in rows:
        class_id = int(row["class_id"])
        image_id = int(row["image_id"])
        if not 1 <= class_id <= 150 or image_id <= 0:
            raise ValueError("Invalid class/image ID in examples CSV")
        if row["label"] and row not in by_class[class_id]:
            by_class[class_id].append(row)
    if sorted(by_class) != list(range(1, 151)):
        raise ValueError("Examples CSV must contain all 150 VG150 classes")
    owner = {}

    def match(class_id, visited):
        for row in by_class[class_id]:
            image_id = int(row["image_id"])
            if image_id in visited:
                continue
            visited.add(image_id)
            if image_id not in owner or match(owner[image_id], visited):
                owner[image_id] = class_id
                return True
        return False

    for class_id in range(1, 151):
        if not match(class_id, set()):
            raise ValueError(f"Cannot choose a distinct train image for class {class_id}")
    selected = {
        class_id: next(row for row in by_class[class_id] if int(row["image_id"]) == image_id)
        for image_id, class_id in owner.items()
    }
    return [selected[class_id] for class_id in range(1, 151)]


def choose_examples(rows, per_class):
    if per_class <= 0:
        raise ValueError("per_class must be positive")
    if per_class == 1:
        return choose_distinct_examples(rows)
    by_class = defaultdict(list)
    for row in rows:
        class_id, image_id = int(row["class_id"]), int(row["image_id"])
        if not 1 <= class_id <= 150 or image_id <= 0:
            raise ValueError("Invalid class/image ID in examples CSV")
        if image_id in {int(item["image_id"]) for item in by_class[class_id]}:
            raise ValueError(f"Repeated image {image_id} for class {class_id}")
        by_class[class_id].append(row)
    if sorted(by_class) != list(range(1, 151)):
        raise ValueError("Examples CSV must contain all 150 VG150 classes")
    if any(len(by_class[class_id]) < per_class for class_id in range(1, 151)):
        raise ValueError(f"Examples CSV has fewer than {per_class} images for a class")
    return [row for class_id in range(1, 151) for row in by_class[class_id][:per_class]]


def archive_member(row):
    image_id = int(row["image_id"])
    parsed = urlparse(row["url"])
    match = re.fullmatch(r"/people/rak248/(VG_100K(?:_2)?)/(\d+)\.jpg", parsed.path)
    if parsed.scheme != "https" or parsed.hostname != "cs.stanford.edu" or not match:
        raise ValueError(f"Unexpected Visual Genome image URL: {row['url']}")
    if int(match.group(2)) != image_id:
        raise ValueError("Example image ID and URL filename disagree")
    return match.group(1), f"{match.group(1)}/{image_id}.jpg"


def check_jpeg(data):
    if not data.startswith(b"\xff\xd8") or not data.endswith(b"\xff\xd9"):
        raise ValueError("Downloaded archive member is not a complete JPEG")


def download_archive(archive_dir, rows, output_dir):
    url = f"{MIRROR}/{ARCHIVES[archive_dir]}"
    results = []
    with zipfile.ZipFile(HTTPRangeFile(url)) as archive:
        for row in rows:
            image_id = int(row["image_id"])
            _, member = archive_member(row)
            info = archive.getinfo(member)
            if info.file_size > 20_000_000:
                raise ValueError(f"Image {image_id} exceeds the 20 MB safety limit")
            target = output_dir / f"{image_id}.jpg"
            if target.exists():
                data = target.read_bytes()
                state = "existing"
            else:
                data = archive.read(info)  # zipfile verifies the member CRC.
                state = "downloaded"
            check_jpeg(data)
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(".jpg.part")
                with temporary.open("xb") as stream:
                    stream.write(data)
                os.replace(temporary, target)
            results.append(
                {
                    "class_id": row["class_id"],
                    "label": row["label"],
                    "image_id": row["image_id"],
                    "object_index": row["object_index"],
                    "box_xyxy": row["box_xyxy"],
                    "original_url": row["url"],
                    "archive": ARCHIVES[archive_dir],
                    "local_path": str(target),
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
            if state == "downloaded" or len(results) % 25 == 0 or len(results) == len(rows):
                print(f"[{len(results)}/{len(rows)} {archive_dir}] {row['label']} ({image_id}) {state}", flush=True)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--examples-csv",
        type=Path,
        default=Path("data/knowledge/level1-human-review-v2/train_examples_with_boxes.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/vg150/images"))
    parser.add_argument(
        "--per-class", type=int, default=5, help="Number of sampled train images per class (default: 5)"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/knowledge/level1-human-review-v2/review_images.csv"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    with args.examples_csv.open(encoding="utf-8-sig", newline="") as stream:
        selected = choose_examples(list(csv.DictReader(stream)), args.per_class)
    groups = defaultdict(list)
    for row in selected:
        archive_dir, _ = archive_member(row)
        groups[archive_dir].append(row)
    print(
        f"Selected {len(selected)} class-image examples ({len({r['image_id'] for r in selected})} unique JPEGs): "
        + ", ".join(f"{key}={len(groups[key])}" for key in sorted(groups))
    )
    if args.dry_run:
        return
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {key: pool.submit(download_archive, key, rows, args.output_dir) for key, rows in groups.items()}
        results = [entry for key in sorted(futures) for entry in futures[key].result()]
    results.sort(key=lambda row: (int(row["class_id"]), int(row["image_id"])))
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.manifest.with_suffix(".csv.part")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(results)
    os.replace(temporary, args.manifest)
    print(
        f"[OK] {len(results)} class-image examples, {len({r['image_id'] for r in results})} JPEGs; manifest: {args.manifest}"
    )


if __name__ == "__main__":
    main()
