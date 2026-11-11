"""Pinned GQA pilot cohorts and verified, frozen region-feature datasets."""

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from src.datasets.gqa import GQAAnnotationLoader
from src.features.geometry import GeometryFeatureExtractor
from src.graph.builder import SceneGraphBuilder
from src.knowledge.transe.core import sha256


def select_cohort(manifest, manifest_digest, train_size=512, val_size=128, seed=42):
    splits = {}
    for split, size in (("train", train_size), ("val", val_size)):
        ids = manifest["splits"][split]
        if not 0 < size <= len(ids) or len(set(ids)) != len(ids):
            raise ValueError("Invalid cohort size or duplicate image IDs")
        splits[split] = sorted(
            ids, key=lambda i: (hashlib.sha256(f"gqa-pilot-v1:{seed}:{split}:{i}".encode()).hexdigest(), i)
        )[:size]
    if set(splits["train"]) & set(splits["val"]):
        raise ValueError("Overlapping pilot splits")
    return {
        "schema_version": 1,
        "dataset": manifest["dataset"],
        "manifest_sha256": manifest_digest,
        "selection": "first N by SHA256(gqa-pilot-v1:seed:split:image_id), without filtering by annotations",
        "seed": seed,
        "splits": splits,
        "test_used": False,
    }


def validate_cohort(cohort, manifest, manifest_digest):
    expected = select_cohort(
        manifest, manifest_digest, len(cohort["splits"]["train"]), len(cohort["splits"]["val"]), cohort["seed"]
    )
    if cohort != expected:
        raise ValueError("Cohort differs from pinned deterministic selection")


def record_identity(record):
    """Bind cache rows to the complete ordered normalized record, including GT."""
    payload = {
        "image_id": record.image_id,
        "size": [record.width, record.height],
        "objects": [[o.object_id, o.label_id, o.bbox_xyxy, o.attribute_ids] for o in record.objects],
        "relations": [[r.subject_id, r.object_id, r.predicate_id] for r in record.relations],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class GQAPilotDataset:
    """Load pilot graphs into CPU memory after verifying source and cache hashes."""

    def __init__(self, root, manifest_path, cohort_path, cache_dir, split):
        if split not in ("train", "val"):
            raise ValueError("Pilot permits only train/val")
        manifest = json.loads(Path(manifest_path).read_text())
        cohort = json.loads(Path(cohort_path).read_text())
        validate_cohort(cohort, manifest, sha256(manifest_path))
        cache_dir = Path(cache_dir)
        metadata = json.loads((cache_dir / "manifest.json").read_text())
        if not metadata["complete"] or metadata["cohort_sha256"] != sha256(cohort_path):
            raise ValueError("Incomplete cache or different cohort")
        if metadata["manifest_sha256"] != sha256(manifest_path):
            raise ValueError("Different source manifest")
        if metadata["features_sha256"] != sha256(cache_dir / "features.h5"):
            raise ValueError("Feature cache hash mismatch")
        loader = GQAAnnotationLoader(root, split, manifest_path)
        index = {image_id: i for i, image_id in enumerate(loader.indices)}
        self.graphs = []
        self.vocabulary = manifest["vocabulary"]
        geometry, builder = GeometryFeatureExtractor(), SceneGraphBuilder()
        with h5py.File(cache_dir / "features.h5", "r") as cache:
            for image_id in cohort["splits"][split]:
                record = loader.load(index[image_id])
                group = cache[image_id]
                if group.attrs["record_sha256"] != record_identity(record):
                    raise ValueError("Cache record/order mismatch")
                vectors = np.asarray(group["features"], dtype=np.float32)
                if vectors.shape != (len(record.objects), 512) or not np.isfinite(vectors).all():
                    raise ValueError("Invalid cached vectors")
                x = torch.cat((geometry.extract(record), torch.from_numpy(vectors)), dim=1)
                self.graphs.append(builder.build(record, x))

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        return self.graphs[index]
