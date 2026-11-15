import csv
import io
import json
import random
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from tools.level3_vietnam_benchmark_v1 import runner as tool


def synthetic_image(seed: int, size=(720, 400)) -> bytes:
    rand = random.Random(seed)
    tile = Image.new("RGB", (9, 8))
    tile.putdata([(rand.randrange(255), rand.randrange(255), rand.randrange(255)) for _ in range(72)])
    image = tile.resize(size, Image.Resampling.NEAREST)
    data = io.BytesIO()
    image.save(data, format="PNG")
    return data.getvalue()


def setup_sources(tmp_path: Path, count: int = 20) -> Path:
    root = tmp_path / "raw"
    for source_index, source in enumerate(("sidewalk", "streetfood", "travel")):
        folder = root / source / "images"
        folder.mkdir(parents=True)
        for index in range(count):
            (folder / f"img-{index:03d}.png").write_bytes(synthetic_image(source_index * 1000 + index))
    return root


def build_args(tmp_path: Path, raw_root: Path, target="pilot"):
    from argparse import Namespace

    return Namespace(
        target=target,
        raw_root=raw_root,
        processed_root=tmp_path / "processed",
        config=Path(tool.__file__).resolve().parents[2] / "configs/level3/vietnam-benchmark-v1.json",
    )


def accept_for_finalization(review_path: Path, quota=10) -> None:
    with review_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    counters = {"sidewalk": 0, "streetfood": 0, "travel": 0}
    for row in rows:
        source = row["dataset"]
        if counters[source] >= quota:
            continue
        counters[source] += 1
        row.update(
            {
                "creator": "Verified Example Author",
                "image_license": "CC BY 4.0",
                "image_license_url": "https://creativecommons.org/licenses/by/4.0/",
                "license_checked": "yes",
                "privacy_clear": "yes",
                "objects_clear": "yes",
                "edit_feasible": "yes",
                "reviewer": "Test reviewer",
                "decision": "accept",
                "notes": "Test data",
            }
        )
        if source == "travel":
            row["original_image_url"] = f"https://commons.wikimedia.org/wiki/File:example_{row['candidate_id']}.png"
    with review_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tool.REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def test_deterministic_pool_is_prefix() -> None:
    names = [f"images/{i:04d}.jpg" for i in range(200)]
    small = tool.select_files(names, "sidewalk", 42, 20)
    large = tool.select_files(names, "sidewalk", 42, 80)
    assert small == large[:20]
    assert small == tool.select_files(names[::-1], "sidewalk", 42, 20)


def test_unsafe_archive_members_rejected() -> None:
    assert not tool.is_safe_path("../../etc/passwd")
    assert not tool.is_safe_path("/absolute/x.jpg")
    assert not tool.is_safe_path("sub/../../x.jpg")
    assert tool.is_safe_path("train/images/photo 001.jpg")


def test_scan_requires_human_approval_and_writes_contact_sheets(tmp_path: Path) -> None:
    raw = setup_sources(tmp_path)
    args = build_args(tmp_path, raw)
    tool.scan(args)
    out = args.processed_root / "pilot"
    data = json.loads((out / "candidate-manifest.json").read_text())
    assert len(data["candidates"]) == 60
    assert all(item["sha256"] for item in data["candidates"])
    assert len(list((out / "contact_sheets").glob("sheet-*.jpg"))) == 3
    with (out / "review.csv").open(newline="", encoding="utf-8") as f:
        reviews = list(csv.DictReader(f))
    assert len(reviews) == 60
    assert all(r["decision"] == "pending" and r["license_checked"] == "" for r in reviews)
    with pytest.raises(ValueError, match="Expected exact quota"):
        tool.finalize(args)


def test_finalize_30_images_and_45_blank_edit_slots(tmp_path: Path) -> None:
    args = build_args(tmp_path, setup_sources(tmp_path))
    tool.scan(args)
    accept_for_finalization(args.processed_root / "pilot/review.csv")
    tool.finalize(args)
    locked = args.processed_root / "pilot/locked"
    manifest = json.loads((locked / "manifest.json").read_text())
    assert manifest["image_count"] == 30
    assert manifest["counts"] == {"sidewalk": 10, "streetfood": 10, "travel": 10}
    assert len(list((locked / "images").rglob("*.png"))) == 30
    with (locked / "edit_slots.csv").open(newline="", encoding="utf-8") as f:
        edits = list(csv.DictReader(f))
    assert len(edits) == 45
    assert all(r["instruction_en"] == "" and r["status"] == "needs_authoring" for r in edits)
    with pytest.raises(FileExistsError):
        tool.finalize(args)


def test_review_cannot_change_image_hash(tmp_path: Path) -> None:
    args = build_args(tmp_path, setup_sources(tmp_path))
    tool.scan(args)
    rev = args.processed_root / "pilot/review.csv"
    accept_for_finalization(rev)
    # Tampering with an actual locked field is rejected.
    with rev.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["sha256"] = "0" * 64
    with rev.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tool.REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="Immutable field modified"):
        tool.finalize(args)


def test_license_is_never_auto_approved(tmp_path: Path) -> None:
    args = build_args(tmp_path, setup_sources(tmp_path))
    tool.scan(args)
    rev = args.processed_root / "pilot/review.csv"
    accept_for_finalization(rev)
    with rev.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    one = next(row for row in rows if row["decision"] == "accept")
    one["license_checked"] = ""
    with rev.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tool.REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="license_checked"):
        tool.finalize(args)


def test_travel_requires_original_wikimedia_url(tmp_path: Path) -> None:
    args = build_args(tmp_path, setup_sources(tmp_path))
    tool.scan(args)
    rev = args.processed_root / "pilot/review.csv"
    accept_for_finalization(rev)
    with rev.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    one = next(row for row in rows if row["decision"] == "accept" and row["dataset"] == "travel")
    one["original_image_url"] = ""
    with rev.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tool.REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="original Wikimedia Commons URL"):
        tool.finalize(args)


def test_zip_reader_ignores_traversal_and_nonimages(tmp_path: Path) -> None:
    path = tmp_path / "food.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("../../bad.png", synthetic_image(1))
        zf.writestr("train/images/ok.png", synthetic_image(2))
        zf.writestr("train/labels/foo.txt", "hello")
    files = list(tool._paths(path))
    assert len(files) == 1
    assert files[0][0] == "train/images/ok.png"


def test_fetch_hf_selects_only_subset_and_pins_revision(tmp_path: Path, monkeypatch) -> None:
    import sys
    import types
    from types import SimpleNamespace

    source_root = tmp_path / "upstream"
    source_root.mkdir()
    names = [f"images/{i:03d}.png" for i in range(50)]
    for index, name in enumerate(names):
        path = source_root / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(synthetic_image(index))
    downloaded = []

    class FakeApi:
        def dataset_info(self, repo_id, revision):
            assert revision == "main"
            return SimpleNamespace(sha="pinned-upstream-sha")

        def list_repo_files(self, repo_id, repo_type, revision):
            assert revision == "pinned-upstream-sha" and repo_type == "dataset"
            return names + ["README.md", "train.csv"]

    def fake_hf_download(repo_id, repo_type, filename, revision):
        assert revision == "pinned-upstream-sha"
        downloaded.append(filename)
        return str(source_root / filename)

    fake = types.ModuleType("huggingface_hub")
    fake.HfApi = FakeApi
    fake.hf_hub_download = fake_hf_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    args = build_args(tmp_path, tmp_path / "raw")
    args.source = "sidewalk"
    args.revision = "main"
    tool.fetch(args)
    assert len(downloaded) == 30  # 10 final pilot images, 3x fetch pool, not all 50.
    receipt = json.loads((args.raw_root / "sidewalk/source-receipt.json").read_text())
    assert receipt["revision_sha"] == "pinned-upstream-sha"
    assert [x["path"] for x in receipt["rows"]] == tool.select_files(names, "sidewalk", 42, 30)
    with pytest.raises(FileExistsError):
        tool.fetch(args)


def test_fetch_streetfood_extracts_only_selected_zip_images(tmp_path: Path, monkeypatch) -> None:
    import sys
    import types
    from types import SimpleNamespace

    archive_path = tmp_path / "streetfood.zip"
    members = [f"dataset/train/images/{index:03d}.png" for index in range(50)]
    with zipfile.ZipFile(archive_path, "w") as archive:
        for idx, name in enumerate(members):
            archive.writestr(name, synthetic_image(idx))
        archive.writestr("dataset/train/labels/train.txt", "nothing")

    class FakeApi:
        def dataset_info(self, repo_id, revision):
            return SimpleNamespace(sha="food-sha")

    def fake_download(repo_id, repo_type, filename, revision):
        assert filename == "Vietnamese_StreetFood_14Class.zip"
        return str(archive_path)

    fake = types.ModuleType("huggingface_hub")
    fake.HfApi = FakeApi
    fake.hf_hub_download = fake_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    args = build_args(tmp_path, tmp_path / "raw")
    args.source = "streetfood"
    args.revision = "main"
    tool.fetch(args)
    receipt = json.loads((args.raw_root / "streetfood/source-receipt.json").read_text())
    assert len(receipt["rows"]) == 30
    assert len(list((args.raw_root / "streetfood").rglob("*.png"))) == 30
    assert not list((args.raw_root / "streetfood").rglob("*.txt"))
