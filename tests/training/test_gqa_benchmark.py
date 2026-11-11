"""Stress selection and cached real-feature alignment safeguards (CPU fixtures)."""

import h5py
import numpy as np
import pytest
import torch

from src.datasets.gqa_pilot import record_identity
from src.datasets.schemas import SceneGraphRecord, SceneObject
from tools.benchmark_gqa_scale import cached_graph, largest_ids


def record(image_id="a", nodes=2):
    return SceneGraphRecord(
        image_id, 100, 100, tuple(SceneObject(i, 1, (i * 10.0, 0.0, i * 10.0 + 5.0, 5.0)) for i in range(nodes))
    )


def test_stress_selection_uses_normalized_node_count_and_stable_id_ties():
    records = {"c": record("c", 3), "b": record("b", 2), "a": record("a", 3)}
    assert largest_ids(records, 2) == ["a", "c"]
    assert largest_ids(dict(reversed(list(records.items()))), 2) == ["a", "c"]
    with pytest.raises(ValueError):
        largest_ids(records, 4)


def test_cached_features_preserve_order_and_reject_reordered_instances(tmp_path):
    r = record()
    vectors = np.stack([np.full(512, 0.25), np.full(512, 0.5)]).astype(np.float16)
    with h5py.File(tmp_path / "features.h5", "w") as f:
        f.attrs["record_sha256"] = record_identity(r)
        f.create_dataset("features", data=vectors)
        g = cached_graph(r, f)
        assert torch.equal(g.x[:, 5:], torch.from_numpy(vectors.astype(np.float32)))
        assert g.x.shape == (2, 517) and g.edge_index.shape == (2, 2)
        reordered = SceneGraphRecord("a", 100, 100, tuple(reversed(r.objects)))
        with pytest.raises(ValueError, match="order mismatch"):
            cached_graph(reordered, f)


@pytest.mark.parametrize("vectors", [np.zeros((1, 512)), np.full((2, 512), np.nan)])
def test_reject_corrupt_cached_vectors(tmp_path, vectors):
    r = record()
    with h5py.File(tmp_path / "features.h5", "w") as f:
        f.attrs["record_sha256"] = record_identity(r)
        f.create_dataset("features", data=vectors)
        with pytest.raises(ValueError, match="Invalid CLIP"):
            cached_graph(r, f)
