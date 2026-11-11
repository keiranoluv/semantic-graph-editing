"""Tests for deterministic VG150 split generation."""

import json

import h5py
import pytest

from scripts.build_vg150_manifest import build_manifest


@pytest.fixture
def manifest_data(tmp_path):
    """Create synthetic metadata and HDF5 image rows."""
    h5_path = tmp_path / "vg.h5"
    metadata_path = tmp_path / "image_data_raw.json"

    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("split", data=[0, 0, 2])
        h5.create_dataset("img_to_first_box", data=[0, 1, 2])
        h5.create_dataset("img_to_first_rel", data=[0, 1, -1])

    metadata = [
        {"image_id": 100, "width": 640, "height": 480},
        {"image_id": 1592, "width": 640, "height": 480},
        {"image_id": 101, "width": 640, "height": 480},
        {"image_id": 102, "width": 640, "height": 480},
    ]
    metadata_path.write_text(json.dumps(metadata))

    return h5_path, metadata_path, tmp_path / "output"


def test_split_generation(manifest_data):
    """Validation takes the first eligible train image."""
    h5_path, metadata_path, output = manifest_data

    manifest = build_manifest(h5_path, metadata_path, output, validation_size=1)

    assert manifest == {
        "train": [1],
        "val": [0],
        "test": [],
    }

    aligned = json.loads((output / "image_data.json").read_text())
    assert [item["image_id"] for item in aligned] == [100, 101, 102]


def test_keep_empty_relations(manifest_data):
    """Allow test images without relationships when configured."""
    h5_path, metadata_path, output = manifest_data

    manifest = build_manifest(
        h5_path,
        metadata_path,
        output,
        validation_size=1,
        filter_empty_relations=False,
    )

    assert manifest["test"] == [2]


def test_invalid_metadata_alignment(manifest_data):
    """Reject metadata that cannot align to the HDF5 rows."""
    h5_path, metadata_path, output = manifest_data

    metadata_path.write_text(json.dumps([{"image_id": 999, "width": 640, "height": 480}]))

    with pytest.raises(ValueError, match="mismatch"):
        build_manifest(h5_path, metadata_path, output, validation_size=1)
