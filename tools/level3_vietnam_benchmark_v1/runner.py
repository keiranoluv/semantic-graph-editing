"""Human-gated VN100 benchmark. No model, no network in offline stages.

Stages:
  fetch:   explicit online Hugging Face download for one source
  scan:    technical inspection, deterministic candidate cohort, contact sheets
  finalize: human-approved, license-verified, privacy-checked assets only

The resulting images/records remain under data/raw or data/processed, not Git.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/level3/vietnam-benchmark-v1.json"
EXTS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_STATUS = {"accept", "reject", "pending"}
REVIEW_COLUMNS = [
    "candidate_id",
    "dataset",
    "relative_path",
    "sha256",
    "width",
    "height",
    "dhash",
    "source_dataset_url",
    "source_asset_url",
    "original_image_url",
    "creator",
    "image_license",
    "image_license_url",
    "license_checked",
    "privacy_clear",
    "objects_clear",
    "edit_feasible",
    "reviewer",
    "decision",
    "notes",
]
EDIT_TYPES = ("attribute", "add_object", "remove_object", "replace_object", "spatial_relation")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_rank(seed: int, source: str, path: str) -> str:
    return digest(f"{seed}|{source}|{path}".encode("utf-8"))


def is_safe_path(raw: str) -> bool:
    name = PurePosixPath(raw.replace("\\", "/"))
    return not name.is_absolute() and bool(name.parts) and not any(part in {"..", "."} for part in name.parts)


def get_config(path: Path) -> dict[str, Any]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if set(cfg["final"]) != {"sidewalk", "streetfood", "travel"}:
        raise ValueError("Invalid final source set")
    if sum(cfg["final"].values()) != 100 or sum(cfg["pilots"].values()) != 30:
        raise ValueError("Invalid dataset quotas")
    return cfg


def write_new(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)


def write_json_new(path: Path, payload: Any) -> None:
    write_new(path, (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"))


def _read_image(data: bytes, quality: dict[str, int]) -> dict[str, Any]:
    if len(data) > quality["max_image_bytes"]:
        raise ValueError("image-too-large-bytes")
    try:
        with Image.open(io.BytesIO(data)) as raw:
            raw.verify()
        with Image.open(io.BytesIO(data)) as raw:
            w, h = raw.size
            if w * h > quality["max_megapixels"] * 1_000_000:
                raise ValueError("image-too-large-pixels")
            if min(w, h) < quality["min_short_side"] or max(w, h) < quality["min_long_side"]:
                raise ValueError("image-too-small")
            # dHash detects near-exact resizes/re-encodes; human checks scene near-duplicates.
            gray = ImageOps.exif_transpose(raw).convert("L").resize((9, 8), Image.Resampling.LANCZOS)
            pix = list(gray.tobytes())
            bits = 0
            for y in range(8):
                for x in range(8):
                    bits = (bits << 1) | int(pix[y * 9 + x] > pix[y * 9 + x + 1])
            return {"sha256": digest(data), "width": w, "height": h, "dhash": f"{bits:016x}"}
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ValueError("invalid-image") from exc


def _paths(root: Path):
    if root.is_dir():
        for path in sorted(root.rglob("*")):
            if path.is_file() and not path.is_symlink() and path.suffix.lower() in EXTS:
                rel = path.relative_to(root).as_posix()
                if is_safe_path(rel):
                    yield rel, path.read_bytes()
    elif root.suffix.lower() == ".zip" and root.is_file():
        with zipfile.ZipFile(root) as archive:
            for entry in sorted(archive.infolist(), key=lambda x: x.filename):
                if entry.is_dir() or not is_safe_path(entry.filename):
                    continue
                if PurePosixPath(entry.filename).suffix.lower() not in EXTS:
                    continue
                if entry.file_size > 20_000_000 or entry.compress_size == 0:
                    continue
                if entry.file_size > 500 * max(entry.compress_size, 1):
                    continue
                with archive.open(entry, "r") as handle:
                    yield entry.filename, handle.read(20_000_001)
    else:
        raise ValueError(f"Missing source root or unsupported input: {root}")


def _load_receipt(root: Path) -> dict:
    receipt = (root if root.is_dir() else root.parent) / "source-receipt.json"
    if receipt.exists():
        return json.loads(receipt.read_text(encoding="utf-8"))
    return {}


def select_files(paths: list[str], source: str, seed: int, count: int) -> list[str]:
    valid = {p for p in paths if is_safe_path(p) and PurePosixPath(p).suffix.lower() in EXTS}
    return sorted(valid, key=lambda x: (stable_rank(seed, source, x), x))[:count]


def fetch(args: argparse.Namespace) -> None:
    """Actual network access only here; download selected images, never entire image repo."""
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Install requirements-level3-vietnam-v1.txt for fetch") from exc

    cfg = get_config(args.config)
    source = args.source
    entry = cfg["sources"][source]
    target_count = cfg["pilots" if args.target == "pilot" else "final"][source]
    requested = target_count * cfg["download_multiplier"]
    root = args.raw_root / source
    receipt_file = root / "source-receipt.json"
    if receipt_file.exists():
        raise FileExistsError(f"Existing receipt, use a new versioned raw root: {receipt_file}")
    api = HfApi()
    info = api.dataset_info(repo_id=entry["repo_id"], revision=args.revision)
    sha = info.sha
    rows: list[dict] = []
    if source in {"sidewalk", "travel"}:
        paths = api.list_repo_files(repo_id=entry["repo_id"], repo_type="dataset", revision=sha)
        paths = [p for p in paths if p.startswith("images/")]
        chosen = select_files(paths, source, cfg["selection_seed"], requested)
        if len(chosen) < requested:
            raise RuntimeError(f"Only {len(chosen)} images available, expected {requested}")
        for path in chosen:
            remote = hf_hub_download(repo_id=entry["repo_id"], repo_type="dataset", filename=path, revision=sha)
            destination = root / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if digest(destination.read_bytes()) != digest(Path(remote).read_bytes()):
                    raise ValueError(f"Different content already present: {destination}")
            else:
                shutil.copyfile(remote, destination)
            rows.append({"path": path, "sha256": digest(destination.read_bytes())})
    else:
        local_zip = hf_hub_download(
            repo_id=entry["repo_id"], repo_type="dataset", filename=entry["zip_file"], revision=sha
        )
        with zipfile.ZipFile(local_zip) as archive:
            names = [x.filename for x in archive.infolist() if x.filename.lower().endswith(tuple(EXTS))]
            chosen = select_files(names, source, cfg["selection_seed"], requested)
            if len(chosen) < requested:
                raise RuntimeError(f"Only {len(chosen)} images available, expected {requested}")
            for path in chosen:
                entry_info = archive.getinfo(path)
                if entry_info.file_size > cfg["image_quality"]["max_image_bytes"]:
                    raise ValueError(f"Unexpected oversized ZIP member: {path}")
                with archive.open(entry_info) as stream:
                    content = stream.read(cfg["image_quality"]["max_image_bytes"] + 1)
                destination = root / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    if digest(destination.read_bytes()) != digest(content):
                        raise ValueError(f"Different content already present: {destination}")
                else:
                    destination.write_bytes(content)
                rows.append({"path": path, "sha256": digest(content)})
    receipt = {
        "source": source,
        "target": args.target,
        "repo_id": entry["repo_id"],
        "revision_sha": sha,
        "dataset_page": entry["source_page"],
        "license_claim": entry["dataset_license_claim"],
        "seed": cfg["selection_seed"],
        "rows": rows,
    }
    write_json_new(receipt_file, receipt)
    print(f"Fetched {len(rows)} candidate images for {source} into {root}; pinned SHA {sha}")


def scan(args: argparse.Namespace) -> None:
    cfg = get_config(args.config)
    quota = cfg["pilots" if args.target == "pilot" else "final"]
    dest = args.processed_root / args.target
    if dest.joinpath("candidate-manifest.json").exists():
        raise FileExistsError("Immutable candidate manifest already exists; use new processed-root or target")
    candidates = []
    diagnostics = {}
    seen_hashes = set()
    for source in ("sidewalk", "streetfood", "travel"):
        root = args.raw_root / source
        entries = []
        rejected = Counter()
        origin = cfg["sources"][source]
        receipt = _load_receipt(root)
        # Guard against reviewing an outdated cohort if source receipt target differs.
        if receipt and receipt.get("target") != args.target:
            raise ValueError(f"Downloaded {source} with target={receipt.get('target')}; use matching target/root")
        receipt_items = {x["path"]: x["sha256"] for x in receipt.get("rows", [])}
        for rel, content in _paths(root):
            if receipt and rel not in receipt_items:
                raise ValueError(f"Untracked file in a pinned source: {source}/{rel}")
            if rel in receipt_items and digest(content) != receipt_items[rel]:
                raise ValueError(f"Upstream cached image modified: {source}/{rel}")
            try:
                technical = _read_image(content, cfg["image_quality"])
            except ValueError as exc:
                rejected[str(exc)] += 1
                continue
            if technical["sha256"] in seen_hashes:
                rejected["exact-duplicate-image"] += 1
                continue
            seen_hashes.add(technical["sha256"])
            cid = f"{source}-{digest((source + '|' + rel).encode())[:14]}"
            source_asset_url = (
                f"https://huggingface.co/datasets/{origin['repo_id']}/blob/"
                f"{receipt.get('revision_sha', 'main')}/"
                f"{origin.get('zip_file', rel)}"
            )
            entries.append(
                {
                    "candidate_id": cid,
                    "dataset": source,
                    "relative_path": rel,
                    "local_path": str((root / rel).resolve()),
                    "source_dataset_url": origin["source_page"],
                    "source_asset_url": source_asset_url,
                    "source_revision_sha": receipt.get("revision_sha"),
                    "license_claim": origin["dataset_license_claim"],
                    **technical,
                }
            )
        entries.sort(key=lambda x: (stable_rank(cfg["selection_seed"], source, x["relative_path"]), x["relative_path"]))
        pool_count = quota[source] * cfg["candidate_multiplier"]
        candidates.extend(entries[:pool_count])
        diagnostics[source] = {
            "valid": len(entries),
            "rejected": dict(rejected),
            "candidates": min(len(entries), pool_count),
        }
        if len(entries) < pool_count:
            raise ValueError(f"Only {len(entries)} technically eligible {source} images, need {pool_count}; fetch more")
    payload = {
        "protocol": cfg["protocol"],
        "target": args.target,
        "seed": cfg["selection_seed"],
        "quotas": quota,
        "raw_root": str(args.raw_root.resolve()),
        "candidates": candidates,
        "diagnostics": diagnostics,
    }
    write_json_new(dest / "candidate-manifest.json", payload)
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / "review.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        for row in candidates:
            writer.writerow(
                {
                    "candidate_id": row["candidate_id"],
                    "dataset": row["dataset"],
                    "relative_path": row["relative_path"],
                    "sha256": row["sha256"],
                    "width": row["width"],
                    "height": row["height"],
                    "dhash": row["dhash"],
                    "source_dataset_url": row["source_dataset_url"],
                    "source_asset_url": row["source_asset_url"],
                    "decision": "pending",
                    "notes": row["license_claim"] + " (dataset claim; verify per asset)",
                }
            )
    make_contact_sheets(candidates, dest / "contact_sheets")
    print(f"Prepared {len(candidates)} candidates and review.csv at {dest}; no images are approved")


def make_contact_sheets(candidates: list[dict], dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for group_index in range(0, len(candidates), 20):
        group = candidates[group_index : group_index + 20]
        sheet = Image.new("RGB", (1000, 5 * 168), "white")
        draw = ImageDraw.Draw(sheet)
        for index, c in enumerate(group):
            col, row = index % 4, index // 4
            try:
                with Image.open(c["local_path"]) as image:
                    image = ImageOps.exif_transpose(image).convert("RGB")
                    thumb = ImageOps.contain(image, (242, 136))
                    x = col * 250 + (250 - thumb.width) // 2
                    y = row * 168 + 2
                    sheet.paste(thumb, (x, y))
            except OSError:
                pass
            draw.text((col * 250 + 3, row * 168 + 142), c["candidate_id"][:24], fill="black")
        sheet.save(dest / f"sheet-{1 + group_index // 20:02d}.jpg", quality=85)


def _read_reviews(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not set(REVIEW_COLUMNS).issubset(set(reader.fieldnames or [])):
            raise ValueError("Review file is missing required columns")
        return list(reader)


def _check_accepted(row: dict) -> None:
    for field in ("license_checked", "privacy_clear", "objects_clear", "edit_feasible"):
        if row[field].strip().lower() != "yes":
            raise ValueError(f"Accepted {row['candidate_id']} lacks {field}=yes")
    for field in ("reviewer", "creator", "image_license", "image_license_url", "source_asset_url"):
        if not row[field].strip():
            raise ValueError(f"Accepted {row['candidate_id']} lacks {field}")
    if row["dataset"] == "travel" and not row["original_image_url"].strip():
        raise ValueError(f"Travel {row['candidate_id']} requires the original Wikimedia Commons URL")
    # No automatic permission inferred from the overall dataset card.
    lic = row["image_license"].strip().upper().replace(" ", "")
    allowed = {"CC0", "CCBY4.0", "CCBY-SA4.0", "CCBY-NC4.0", "CCBY-NC-SA4.0"}
    if lic not in allowed:
        raise ValueError(f"Unsupported/unverified license format for {row['candidate_id']}: {lic}")


def hamming_hex(a: str, b: str) -> int:
    return (int(a, 16) ^ int(b, 16)).bit_count()


def finalize(args: argparse.Namespace) -> None:
    cfg = get_config(args.config)
    quota = cfg["pilots" if args.target == "pilot" else "final"]
    dest = args.processed_root / args.target
    source = json.loads((dest / "candidate-manifest.json").read_text(encoding="utf-8"))
    if source["protocol"] != cfg["protocol"] or source["target"] != args.target:
        raise ValueError("Manifest/config/target mismatch")
    reference = {x["candidate_id"]: x for x in source["candidates"]}
    reviews = _read_reviews(dest / "review.csv")
    if len(reviews) != len(reference):
        raise ValueError("Review must contain exactly one row per candidate")
    accepted: list[dict] = []
    reviewed_ids = set()
    for row in reviews:
        cid = row["candidate_id"]
        if cid not in reference or cid in reviewed_ids:
            raise ValueError(f"Unknown or duplicate review candidate ID: {cid}")
        reviewed_ids.add(cid)
        original = reference[cid]
        for k in (
            "dataset",
            "relative_path",
            "sha256",
            "dhash",
            "width",
            "height",
            "source_dataset_url",
            "source_asset_url",
        ):
            if str(row[k]).strip() != str(original[k]):
                raise ValueError(f"Immutable field modified for {cid}: {k}")
        if row["decision"].strip().lower() not in ALLOWED_STATUS:
            raise ValueError("Unsupported review decision")
        if row["decision"].strip().lower() == "accept":
            _check_accepted(row)
            content = Path(original["local_path"]).read_bytes()
            if digest(content) != original["sha256"]:
                raise ValueError(f"Source bytes changed after scan: {cid}")
            accepted.append({**original, "human_review": row})
    counts = Counter(x["dataset"] for x in accepted)
    if dict(counts) != quota:
        raise ValueError(f"Expected exact quota {quota}, accepted {dict(counts)}; no export")
    if len({x["sha256"] for x in accepted}) != len(accepted):
        raise ValueError("Exact duplicates among accepted assets")
    for idx, first in enumerate(accepted):
        for second in accepted[idx + 1 :]:
            if hamming_hex(first["dhash"], second["dhash"]) <= 4:
                raise ValueError(f"Near duplicate images: {first['candidate_id']}, {second['candidate_id']}")
    final_dir = dest / "locked"
    if final_dir.exists():
        raise FileExistsError(f"Immutable output already exists: {final_dir}")
    final_dir.mkdir(parents=True)
    rows = []
    for rec in sorted(accepted, key=lambda x: x["candidate_id"]):
        image_rel = f"images/{rec['dataset']}/{rec['candidate_id']}{Path(rec['relative_path']).suffix.lower()}"
        target = final_dir / image_rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(rec["local_path"], target)
        if digest(target.read_bytes()) != rec["sha256"]:
            raise ValueError(f"Copy mismatch: {rec['candidate_id']}")
        human = rec["human_review"]
        rows.append(
            {
                "image_id": rec["candidate_id"],
                "image_path": image_rel,
                "sha256": rec["sha256"],
                "dhash": rec["dhash"],
                "width": rec["width"],
                "height": rec["height"],
                "dataset": rec["dataset"],
                "source_dataset_url": rec["source_dataset_url"],
                "source_asset_url": rec["source_asset_url"],
                "source_revision_sha": rec["source_revision_sha"],
                "original_image_url": human["original_image_url"],
                "creator": human["creator"],
                "image_license": human["image_license"],
                "image_license_url": human["image_license_url"],
                "reviewer": human["reviewer"],
                "review_notes": human["notes"],
            }
        )
    write_json_new(
        final_dir / "manifest.json",
        {
            "protocol": cfg["protocol"],
            "target": args.target,
            "seed": cfg["selection_seed"],
            "counts": dict(counts),
            "image_count": len(rows),
            "images": rows,
            "dataset_notice": "All selected assets manually reviewed for provenance and privacy; not for commercial use if CC BY-NC.",
        },
    )
    # Do not invent editing instructions; human authoring must use the image content.
    count_two = len(rows) // 2
    ranked = sorted(rows, key=lambda x: stable_rank(cfg["selection_seed"], "edit-slots", x["image_id"]))
    double_ids = {x["image_id"] for x in ranked[:count_two]}
    with (final_dir / "edit_slots.csv").open("x", encoding="utf-8", newline="") as handle:
        fields = ["edit_id", "image_id", "edit_type_suggestion", "instruction_en", "annotator", "status", "notes"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            total = 2 if row["image_id"] in double_ids else 1
            for index in range(total):
                edit_id = f"{row['image_id']}-edit{index + 1}"
                kind = EDIT_TYPES[int(stable_rank(cfg["selection_seed"], "edit", edit_id)[:8], 16) % len(EDIT_TYPES)]
                writer.writerow(
                    {
                        "edit_id": edit_id,
                        "image_id": row["image_id"],
                        "edit_type_suggestion": kind,
                        "status": "needs_authoring",
                        "notes": "Check feasible object/relation and preservation scope",
                    }
                )
    write_json_new(
        final_dir / "data-card-draft.json",
        {
            "name": "Vietnam Scene Graph Editing Benchmark v1",
            "status": "images-approved; instructions-pending",
            "images": len(rows),
            "instruction_slots": len(rows) + count_two,
            "sources": cfg["sources"],
            "collection": "Source photographs selected by humans from 3 upstream datasets.",
            "restrictions": [
                "Record per-image authorship and license",
                "Preserve attribution and ShareAlike when applicable",
                "No commercial redistribution of CC BY-NC assets",
                "Do not upload faces/plates to public services",
                "Check graph identity and manual edit outcomes",
            ],
        },
    )
    print(f"Locked {len(rows)} reviewed images and {len(rows) + count_two} unfilled edit slots at {final_dir}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["fetch", "scan", "finalize"])
    parser.add_argument("--source", choices=["sidewalk", "streetfood", "travel"])
    parser.add_argument("--target", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--raw-root", type=Path, default=ROOT / "data/raw/level3/vietnam-benchmark-v1")
    parser.add_argument("--processed-root", type=Path, default=ROOT / "data/processed/level3/vietnam-benchmark-v1")
    args = parser.parse_args(argv)
    if args.stage == "fetch":
        if not args.source:
            parser.error("fetch requires --source")
        fetch(args)
    elif args.stage == "scan":
        scan(args)
    else:
        finalize(args)


if __name__ == "__main__":
    main()
