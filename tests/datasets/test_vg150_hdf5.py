"""Synthetic HDF5 integration tests for the VG150 annotation reader."""

import json

import h5py
import numpy as np
import pytest

from src.datasets.vg150 import VG150AnnotationLoader


@pytest.fixture
def vg150_root(tmp_path):
    """Create two small, HDF5-aligned synthetic image records."""
    root = tmp_path / "vg150"
    root.mkdir()

    with h5py.File(root / "VG-SGG-with-attri.h5", "w") as h5:
        h5.create_dataset("split", data=np.array([0, 2]))
        h5.create_dataset("img_to_first_box", data=[0, 2])
        h5.create_dataset("img_to_last_box", data=[1, 2])
        h5.create_dataset("img_to_first_rel", data=[0, -1])
        h5.create_dataset("img_to_last_rel", data=[1, -1])

        h5.create_dataset("labels", data=[[1], [2], [3]])
        h5.create_dataset(
            "attributes",
            data=[[5, 0], [0, 0], [7, 0]],
        )

        h5.create_dataset(
            "boxes_1024",
            data=[
                [160, 160, 160, 160],
                [480, 320, 160, 160],
                [512, 512, 256, 256],
            ],
        )

        # Two predicates on the same directed object pair.
        h5.create_dataset("relationships", data=[[0, 1], [0, 1]])
        h5.create_dataset("predicates", data=[[1], [2]])

    metadata = [
        {"image_id": 100, "width": 640, "height": 480},
        {"image_id": 200, "width": 1024, "height": 1024},
    ]
    (root / "image_data.json").write_text(json.dumps(metadata))

    vocabulary = {
        "label_to_idx": {"person": 1, "horse": 2, "grass": 3},
        "predicate_to_idx": {"riding": 1, "on": 2},
        "attribute_to_idx": {"brown": 5, "green": 7},
    }
    (root / "VG-SGG-dicts-with-attri.json").write_text(json.dumps(vocabulary))

    manifest = {
        "train": [0],
        "val": [],
        "test": [1],
    }
    manifest_path = root / "splits.json"
    manifest_path.write_text(json.dumps(manifest))

    return root, manifest_path


def test_train_record(vg150_root):
    """Load object labels and image information."""
    root, manifest = vg150_root
    loader = VG150AnnotationLoader(root, "train", manifest)

    assert len(loader) == 1
    record = loader.load(0)

    assert record.image_id == "100"
    assert len(record.objects) == 2
    assert [obj.label_id for obj in record.objects] == [1, 2]
    assert record.objects[0].attribute_ids == (5,)
    record.validate()


def test_bounding_box_conversion(vg150_root):
    """Decode 1024-scale center boxes into pixel-space xyxy."""
    root, manifest = vg150_root
    record = VG150AnnotationLoader(root, "train", manifest).load(0)

    # Original box [160,160,160,160] becomes
    # [80,80,240,240] at scale 1024.
    # Original image max dimension is 640.
    np.testing.assert_allclose(
        record.objects[0].bbox_xyxy,
        [50.0, 50.0, 150.0, 150.0],
    )


def test_multiple_predicates_preserved(vg150_root):
    """Do not collapse multiple predicates on one subject-object pair."""
    root, manifest = vg150_root
    record = VG150AnnotationLoader(root, "train", manifest).load(0)

    assert len(record.relations) == 2
    assert [(r.subject_id, r.object_id, r.predicate_id) for r in record.relations] == [(0, 1, 1), (0, 1, 2)]


def test_empty_relationships(vg150_root):
    """An image may have objects but no annotated relationships."""
    root, manifest = vg150_root
    record = VG150AnnotationLoader(root, "test", manifest).load(0)

    assert record.image_id == "200"
    assert len(record.objects) == 1
    assert record.relations == ()


def test_invalid_split_overlap(vg150_root):
    """Reject overlapping image-level splits."""
    root, manifest = vg150_root
    manifest.write_text(json.dumps({"train": [0], "val": [0], "test": [1]}))

    with pytest.raises(ValueError, match="disjoint"):
        VG150AnnotationLoader(root, "train", manifest)


def test_invalid_split_flag(vg150_root):
    """Reject a split manifest conflicting with HDF5 flags."""
    root, manifest = vg150_root
    manifest.write_text(json.dumps({"train": [1], "val": [], "test": [0]}))

    with pytest.raises(ValueError, match="split flag"):
        VG150AnnotationLoader(root, "train", manifest)
