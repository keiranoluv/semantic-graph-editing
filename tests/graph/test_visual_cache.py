"""Synthetic cache checks; no image encoder or GPU is needed."""

import json

import h5py
import numpy as np
import pytest
import torch

from src.datasets.schemas import SceneGraphRecord, SceneObject
from src.features.visual import GeometryVisualFeatureExtractor, VisualFeatureExtractor
from src.knowledge.transe.core import sha256


def make_cache(tmp_path):
    root = tmp_path / "raw"
    root.mkdir()
    (root / "splits.json").write_text("{}")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache_path = cache_dir / "features.h5"
    with h5py.File(cache_path, "w") as cache:
        cache.create_dataset("done", data=[True])
        cache.create_dataset("image_id", data=[123])
        cache.create_dataset("row_first", data=[0])
        cache.create_dataset("row_last", data=[1])
        cache.create_dataset("boxes_xyxy", data=[[1, 2, 5, 6], [3, 4, 8, 9]])
        cache.create_dataset("features", data=np.arange(1024, dtype=np.float16).reshape(2, 512))
    manifest = {
        "schema_version": 1,
        "complete": True,
        "box_policy": "clip",
        "visual_dim": 512,
        "source_sha256": {"splits.json": sha256(root / "splits.json")},
        "features_sha256": sha256(cache_path),
    }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest))
    record = SceneGraphRecord(
        image_id="123",
        width=10,
        height=10,
        objects=(SceneObject(0, 1, (1, 2, 5, 6)), SceneObject(1, 2, (3, 4, 8, 9))),
        metadata={"hdf5_row": "0", "box_policy": "clip"},
    )
    return root, cache_dir, record


def test_visual_cache_preserves_gt_box_order(tmp_path):
    root, cache_dir, record = make_cache(tmp_path)
    extractor = VisualFeatureExtractor(cache_dir, root)
    visual = extractor.extract(record)
    assert visual.shape == (2, 512)
    assert visual.dtype == torch.float32
    assert visual[0, 0] == 0 and visual[1, 0] == 512
    combined = GeometryVisualFeatureExtractor(extractor).extract(record)
    assert combined.shape == (2, 517)
    assert torch.equal(combined[:, 5:], visual)


def test_visual_cache_rejects_changed_box_and_source(tmp_path):
    root, cache_dir, record = make_cache(tmp_path)
    extractor = VisualFeatureExtractor(cache_dir, root)
    wrong = SceneGraphRecord(
        image_id=record.image_id,
        width=10,
        height=10,
        objects=(SceneObject(0, 1, (1, 2, 6, 6)), record.objects[1]),
        metadata=record.metadata,
    )
    with pytest.raises(ValueError, match="GT boxes mismatch"):
        extractor.extract(wrong)
    (root / "splits.json").write_text('{"changed": true}')
    with pytest.raises(ValueError, match="source hash mismatch"):
        VisualFeatureExtractor(cache_dir, root)
