"""Download and verify every JPEG used by the pinned VG150 image split."""

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

from PIL import Image

from scripts.download_level1_review_images import archive_member

# JPEGs are original Visual Genome images. The Stanford image host was
# unavailable from this machine, so the full archives come from this mirror.
# Keep the concrete URLs here so the download source is visible in this script.
ARCHIVES = {"VG_100K": "images.zip", "VG_100K_2": "images2.zip"}
ARCHIVE_URLS = {
    "VG_100K": "https://huggingface.co/datasets/pppop7/VisualGenome_VG_100K_1_and_2/resolve/main/images.zip",
    "VG_100K_2": "https://huggingface.co/datasets/pppop7/VisualGenome_VG_100K_1_and_2/resolve/main/images2.zip",
}


def required_images(root: Path) -> dict[str, dict]:
    split = json.loads((root / "splits.json").read_text(encoding="utf-8"))
    metadata = json.loads((root / "image_data.json").read_text(encoding="utf-8"))
    rows = sorted(set(split["train"]) | set(split["val"]) | set(split["test"]))
    required = {}
    for row in rows:
        info = metadata[row]
        image_id = str(info["image_id"])
        if image_id in required:
            raise ValueError(f"Duplicate image ID in split: {image_id}")
        archive, member = archive_member(info)
        required[image_id] = {"row": row, "archive": archive, "member": member, "info": info}
    return required


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_jpeg(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        if image.format != "JPEG":
            raise ValueError(f"Not a JPEG: {path}")
        size = image.size
        image.verify()
    return size


def download_archive(archive_path: Path, archive_dir: str) -> None:
    if archive_path.is_file() and zipfile.is_zipfile(archive_path):
        return
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    partial = archive_path.with_suffix(".zip.part")
    url = ARCHIVE_URLS[archive_dir]
    subprocess.run(
        ["curl", "-fsSL", "--retry", "5", "--retry-all-errors", "-C", "-", "-o", str(partial), url],
        check=True,
    )
    if not zipfile.is_zipfile(partial):
        raise ValueError(f"Incomplete or invalid ZIP: {partial}")
    partial.replace(archive_path)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/vg150"))
    parser.add_argument("--archive-dir", type=Path, default=Path("data/raw/vg150/archives"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    required = required_images(args.root)
    groups = {key: [image_id for image_id, item in required.items() if item["archive"] == key] for key in ARCHIVES}
    print(f"[PLAN] {len(required)} JPEGs: " + ", ".join(f"{key}={len(groups[key])}" for key in ARCHIVES), flush=True)
    for archive_dir, url in ARCHIVE_URLS.items():
        print(f"[SOURCE] {archive_dir}: {url}", flush=True)
    if args.dry_run:
        return
    output = args.root / "images"
    output.mkdir(parents=True, exist_ok=True)
    archive_hashes = {}
    for archive_dir, archive_name in ARCHIVES.items():
        archive_path = args.archive_dir / archive_name
        if not args.verify_only:
            download_archive(archive_path, archive_dir)
        if archive_path.is_file():
            archive_hashes[archive_name] = sha256(archive_path)
        elif not args.verify_only:
            raise FileNotFoundError(archive_path)
        if args.verify_only:
            continue
        with zipfile.ZipFile(archive_path) as archive:
            for count, image_id in enumerate(groups[archive_dir], start=1):
                item = required[image_id]
                target = output / f"{image_id}.jpg"
                if not target.is_file():
                    source = archive.getinfo(item["member"])
                    if source.file_size > 20_000_000:
                        raise ValueError(f"JPEG exceeds 20 MB: {item['member']}")
                    temporary = target.with_suffix(".jpg.part")
                    with archive.open(source) as reader, temporary.open("xb") as writer:
                        for chunk in iter(lambda: reader.read(1024 * 1024), b""):
                            writer.write(chunk)
                    temporary.replace(target)
                verify_jpeg(target)
                if count % 1000 == 0 or count == len(groups[archive_dir]):
                    print(f"[{archive_dir}] {count}/{len(groups[archive_dir])}", flush=True)
    image_rows = []
    for count, (image_id, item) in enumerate(sorted(required.items(), key=lambda pair: pair[1]["row"]), start=1):
        path = output / f"{image_id}.jpg"
        if not path.is_file():
            raise FileNotFoundError(path)
        width, height = verify_jpeg(path)
        image_rows.append(
            {
                "row": item["row"],
                "image_id": image_id,
                "archive": ARCHIVES[item["archive"]],
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "width": width,
                "height": height,
                "metadata_width": int(item["info"]["width"]),
                "metadata_height": int(item["info"]["height"]),
            }
        )
        if count % 5000 == 0 or count == len(required):
            print(f"[VERIFY] {count}/{len(required)}", flush=True)
    manifest = {
        "schema_version": 1,
        "images": image_rows,
        "archive_sha256": archive_hashes,
        "split_sha256": sha256(args.root / "splits.json"),
        "metadata_sha256": sha256(args.root / "image_data.json"),
    }
    manifest_path = args.root / "images_manifest.json"
    temporary = manifest_path.with_suffix(".json.part")
    temporary.write_text(json.dumps(manifest, separators=(",", ":")) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    mismatch = sum(
        (row["width"], row["height"]) != (row["metadata_width"], row["metadata_height"]) for row in image_rows
    )
    print(f"[OK] {len(image_rows)} JPEGs verified; dimension_mismatches={mismatch}; manifest={manifest_path}")


if __name__ == "__main__":
    main()
