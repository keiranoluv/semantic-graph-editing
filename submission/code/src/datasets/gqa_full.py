"""Atomic sharded CLIP cache and lazy graphs for the full GQA experiment."""

import hashlib
import io
import json
import os
import zipfile
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

from src.datasets.gqa import image_archive_index, normalize_gqa
from src.datasets.gqa_pilot import record_identity, select_cohort
from src.datasets.schemas import SceneGraphRecord, SceneObject, SceneRelation
from src.features.geometry import GeometryFeatureExtractor
from src.graph.builder import SceneGraphBuilder
from src.knowledge.transe.core import sha256


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def pin_json(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Pinned identity differs: {path}; use a new version/directory")
    else:
        atomic_write(path, value)


def cohort_for(config, manifest):
    groups = manifest["splits"]
    if any(len(g) != len(set(g)) for g in groups.values()) or any(
        set(groups[a]) & set(groups[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
    ):
        raise ValueError("Non-disjoint or duplicate split")
    if config["scope"] == "full":
        return {k: list(groups[k]) for k in ("train", "val")}
    if config["scope"] != "smoke":
        raise ValueError("Unknown scope")
    return select_cohort(manifest, "unused", config["smoke_train_size"], config["smoke_val_size"], 42)["splits"]


def unpack_record(payload):
    r = json.loads(payload)
    objects = tuple(
        SceneObject(o["object_id"], o["label_id"], tuple(o["bbox_xyxy"]), tuple(o["attribute_ids"]))
        for o in r["objects"]
    )
    return SceneGraphRecord(
        r["image_id"],
        r["width"],
        r["height"],
        objects,
        tuple(SceneRelation(**relation) for relation in r["relations"]),
        r["metadata"],
    )


def read_group(group):
    record = unpack_record(group["record"][()].decode())
    record.validate()
    if record_identity(record) != group.attrs["record_sha256"]:
        raise ValueError("Record identity mismatch")
    features = np.asarray(group["features"], dtype=np.float32)
    if features.shape != (len(record.objects), 512) or not np.isfinite(features).all():
        raise ValueError("Invalid cached feature shape/values")
    return record, features


def cache_identity(config, manifest, cohort):
    return {
        "schema_version": 1,
        "scope": config["scope"],
        "splits": cohort,
        "manifest_sha256": sha256(Path(config["manifest"])),
        "annotation_sha256": manifest["annotation_sha256"]["train_sceneGraphs.json"],
        "image_archive_sha256": sha256(Path(config["root"]) / "archives/images.zip"),
        "weights_sha256": sha256(Path(config["weights"])),
        "shard_size": config["cache_shard_size"],
        "clip_batch_size": config["clip_batch_size"],
        "device": config["device"],
        "torch_version": str(torch.__version__),
        "encoder": "frozen OpenAI CLIP ViT-B/32; default transform; normalized float16",
        "box_policy": "GT xyxy scaled to JPEG, floor/ceil crop",
        "source_sha256": {
            name: sha256(Path(name))
            for name in (
                "src/datasets/gqa_full.py",
                "src/datasets/gqa.py",
                "src/datasets/gqa_pilot.py",
                "scripts/cache_vg150_regions.py",
            )
        },
        "test_used": False,
    }


def shard_plan(identity):
    plans = []
    for split, ids in identity["splits"].items():
        for start in range(0, len(ids), identity["shard_size"]):
            plans.append(
                (f"{split}-{start // identity['shard_size']:05d}.h5", ids[start : start + identity["shard_size"]])
            )
    return plans


def prepare_cache(config, *, stop_requested=lambda: False):
    """Completed shards are immutable; interrupted temp shard is rebuilt on resume."""
    import clip

    from scripts.cache_vg150_regions import extract_vectors, scale_boxes_to_jpeg

    manifest = json.loads(Path(config["manifest"]).read_text())
    cohort = cohort_for(config, manifest)
    directory = Path(config["cache"])
    directory.mkdir(parents=True, exist_ok=True)
    identity = cache_identity(config, manifest, cohort)
    pin_json(directory / "identity.json", identity)
    if (directory / "manifest.json").exists():
        verify_cache(directory, identity)
        print("[CACHE] completed cache verified", flush=True)
        return True
    annotation = Path(config["root"]) / "sceneGraphs/train_sceneGraphs.json"
    if sha256(annotation) != identity["annotation_sha256"]:
        raise ValueError("Training annotation changed")
    raw = json.loads(annotation.read_text())
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


def verify_cache(directory, expected_identity=None):
    directory = Path(directory)
    metadata = json.loads((directory / "manifest.json").read_text())
    if not metadata["complete"] or (expected_identity is not None and metadata["identity"] != expected_identity):
        raise ValueError("Incomplete/different cache")
    plan = shard_plan(metadata["identity"])
    if len(metadata["shards"]) != len(plan):
        raise ValueError("Missing cache shards")
    for entry, (name, ids) in zip(metadata["shards"], plan, strict=True):
        if entry["file"] != name or entry["ids"] != ids or entry["identity_sha256"] != digest(metadata["identity"]):
            raise ValueError("Incorrect shard index")
        if sha256(directory / name) != entry["sha256"]:
            raise ValueError(f"Cache shard hash mismatch: {name}")
    return metadata


class GQAFullDataset:
    """Lazy graph construction with bounded HDF5 handles; no full pair tensors in RAM."""

    def __init__(self, directory, metadata, split):
        if split not in ("train", "val"):
            raise ValueError("This experiment only loads train/validation")
        self.directory = Path(directory)
        self.ids = metadata["identity"]["splits"][split]
        self.locations = {i: e["file"] for e in metadata["shards"] for i in e["ids"]}
        self.handles = OrderedDict()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        image_id = self.ids[index]
        name = self.locations[image_id]
        if name not in self.handles:
            if len(self.handles) >= 8:
                self.handles.popitem(last=False)[1].close()
            self.handles[name] = h5py.File(self.directory / name, "r")
        self.handles.move_to_end(name)
        record, vectors = read_group(self.handles[name][image_id])
        if record.image_id != image_id:
            raise ValueError("Cache image order changed")
        x = torch.cat((GeometryFeatureExtractor().extract(record), torch.from_numpy(vectors)), dim=1)
        return SceneGraphBuilder().build(record, x)

    def close(self):
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()
