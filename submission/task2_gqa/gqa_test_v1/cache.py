"""Held-out cache, adapted from the frozen training cache without modifying it."""
import hashlib
import io
import json
import os
import zipfile
from collections import Counter, OrderedDict
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from common.datasets.gqa import image_archive_index, normalize_gqa
from common.datasets.gqa_full import (
    GQAFullDataset,
    atomic_write,
    cache_identity,
    cohort_for,
    digest,
    pin_json,
    read_group,
    shard_plan,
    verify_cache,
)
from common.datasets.gqa_pilot import record_identity
from common.knowledge.transe.core import sha256


def test_cache_identity(config, manifest):
    cohort_for({"scope": "full"}, manifest)  # verify all three splits are disjoint
    identity = cache_identity(config, manifest, {"test": manifest["splits"]["test"]})
    identity["scope"] = "project-test-v1"
    identity["encoder_environment"] = {name: version(name) for name in ("torchvision", "clip", "Pillow", "numpy")}
    if config.get("reference_cache"):
        reference = json.loads((Path(config["reference_cache"]) / "manifest.json").read_text())["identity"]
        for key in ("manifest_sha256", "weights_sha256", "image_archive_sha256", "encoder", "box_policy"):
            if identity[key] != reference[key]:
                raise ValueError(f"Test/train feature provenance differs: {key}")
    identity["test_used"] = True
    identity["annotation_sha256"] = manifest["annotation_sha256"]["val_sceneGraphs.json"]
    identity["source_sha256"][__file__.replace(str(Path.cwd()) + "/", "")] = sha256(Path(__file__))
    return identity


class TestDataset(GQAFullDataset):
    def __init__(self, directory, metadata):
        if set(metadata["identity"]["splits"]) != {"test"}:
            raise ValueError("Expected test-only cache")
        self.directory = Path(directory)
        self.ids = metadata["identity"]["splits"]["test"]
        self.locations = {i: e["file"] for e in metadata["shards"] for i in e["ids"]}
        self.handles = OrderedDict()


def prepare_cache(config, *, stop_requested=lambda: False):
    """Completed shards are immutable; interrupted temp shard is rebuilt on resume."""
    import clip

    from task1_vg150.cache_vg150_regions import extract_vectors, scale_boxes_to_jpeg

    manifest = json.loads(Path(config["manifest"]).read_text())
    directory = Path(config["cache"])
    directory.mkdir(parents=True, exist_ok=True)
    identity = test_cache_identity(config, manifest)
    pin_json(directory / "identity.json", identity)
    if (directory / "manifest.json").exists():
        verify_cache(directory, identity)
        print("[CACHE] completed cache verified", flush=True)
        return True
    annotation = Path(config["root"]) / "sceneGraphs/val_sceneGraphs.json"
    if sha256(annotation) != identity["annotation_sha256"]:
        raise ValueError("Test annotation changed")
    raw = json.loads(annotation.read_text())
    statistics = Counter()
    for image_id in manifest["splits"]["test"]:
        _, counts = normalize_gqa(image_id, raw[image_id], manifest["vocabulary"])
        statistics.update(counts)
    pin_json(directory / "normalization-statistics.json", dict(statistics))
    # Validate completed shards before any GPU work on a resumed cache.
    for name, ids in shard_plan(identity):
        marker = directory / (name + ".json")
        if marker.exists():
            entry = json.loads(marker.read_text())
            if (
                entry["file"] != name
                or entry["ids"] != ids
                or entry["identity_sha256"] != digest(identity)
                or entry["sha256"] != sha256(directory / name)
            ):
                raise ValueError(f"Completed shard changed: {name}")
    archive = Path(config["root"]) / "archives/images.zip"
    image_index = image_archive_index(archive)
    model, transform = None, None
    entries = []
    with zipfile.ZipFile(archive) as images:
        for name, ids in shard_plan(identity):
            if stop_requested():
                return False
            path, marker = directory / name, directory / (name + ".json")
            if marker.exists():
                entry = json.loads(marker.read_text())
                if entry["file"] != name or entry["ids"] != ids or entry["sha256"] != sha256(path):
                    raise ValueError(f"Completed shard changed: {name}")
                if entry["identity_sha256"] != digest(identity):
                    raise ValueError("Shard identity changed")
                entries.append(entry)
                continue
            if not path.exists():
                if model is None:
                    model, transform = clip.load(config["weights"], device=config["device"], jit=False)
                    model.eval().requires_grad_(False)
                temp = path.with_suffix(".h5.tmp")
                with h5py.File(temp, "w") as out:
                    out.attrs["identity_sha256"] = digest(identity)
                    for image_id in ids:
                        if stop_requested():
                            return False
                        record = normalize_gqa(image_id, raw[image_id], manifest["vocabulary"])[0]
                        jpeg = images.read(image_index[image_id])
                        with Image.open(io.BytesIO(jpeg)) as im:
                            im = im.convert("RGB")
                            boxes = scale_boxes_to_jpeg(
                                [o.bbox_xyxy for o in record.objects], (record.width, record.height), im.size
                            )
                            features = extract_vectors(
                                model, transform, im, boxes, config["clip_batch_size"], config["device"]
                            )
                        group = out.create_group(image_id)
                        group.create_dataset("features", data=features.astype(np.float16))
                        group.create_dataset("record", data=json.dumps(asdict(record)).encode())
                        group.attrs["record_sha256"] = record_identity(record)
                        group.attrs["jpeg_sha256"] = hashlib.sha256(jpeg).hexdigest()
                with temp.open("rb") as f:
                    os.fsync(f.fileno())
                os.replace(temp, path)
            # Also recover a crash between shard rename and marker publication.
            with h5py.File(path) as f:
                if set(f) != set(ids) or f.attrs["identity_sha256"] != digest(identity):
                    raise ValueError("Uncommitted shard has incorrect identity/IDs")
                for image_id in ids:
                    record, _ = read_group(f[image_id])
                    expected = normalize_gqa(image_id, raw[image_id], manifest["vocabulary"])[0]
                    if record_identity(record) != record_identity(expected):
                        raise ValueError("Shard annotations differ from source")
                    if (
                        f[image_id].attrs["jpeg_sha256"]
                        != hashlib.sha256(images.read(image_index[image_id])).hexdigest()
                    ):
                        raise ValueError("Shard JPEG differs from source")
            entry = {"file": name, "ids": ids, "sha256": sha256(path), "identity_sha256": digest(identity)}
            pin_json(marker, entry)
            entries.append(entry)
            print(f"[CACHE] {len(entries)}/{len(shard_plan(identity))} shards complete: {name}", flush=True)
    atomic_write(directory / "manifest.json", {"identity": identity, "shards": entries, "complete": True})
    return True
