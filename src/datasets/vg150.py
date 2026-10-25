"""VG150 HDF5 annotation reader and PyG dataset adapter."""

import json
from math import isfinite
from pathlib import Path
from typing import Literal

import h5py
from torch_geometric.data import Data, Dataset

from src.datasets.hdf5_session import ProcessLocalH5
from src.datasets.schemas import SceneGraphRecord, SceneObject, SceneRelation
from src.graph.builder import SceneGraphBuilder
from src.graph.features import FeatureExtractor


def _clip_box_to_image(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    """Clip coordinates to image bounds and reject degenerate boxes."""
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")

    if not all(isfinite(value) for value in box):
        raise ValueError("Bounding box coordinates must be finite")

    x1, y1, x2, y2 = box

    if x1 >= x2 or y1 >= y2:
        raise ValueError("Invalid bounding box geometry")

    clipped = (
        max(0.0, min(float(width), x1)),
        max(0.0, min(float(height), y1)),
        max(0.0, min(float(width), x2)),
        max(0.0, min(float(height), y2)),
    )

    cx1, cy1, cx2, cy2 = clipped

    if cx1 >= cx2 or cy1 >= cy2:
        raise ValueError("Degenerate bounding box after clipping")

    return clipped


def _clip_rounding_overflow(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
    tolerance: float,
) -> tuple[float, float, float, float]:
    """Clip only minor boundary overflow caused by coordinate rounding."""
    x1, y1, x2, y2 = box

    overflow = max(
        -x1,
        -y1,
        x2 - width,
        y2 - height,
        0.0,
    )

    if overflow > tolerance:
        raise ValueError(f"Bounding box exceeds image bounds by {overflow:.4f} pixels")

    return (
        max(0.0, x1),
        max(0.0, y1),
        min(float(width), x2),
        min(float(height), y2),
    )


BoxPolicy = Literal["strict", "clip"]
Split = Literal["train", "val", "test"]
BOX_SCALE = 1024.0


class VG150AnnotationLoader:
    """Read normalized scene records from a pinned VG150 HDF5 variant.

    The split manifest stores HDF5 image-row indices, not image IDs.
    Image metadata must be aligned one-to-one with HDF5 image rows.
    No implicit filtering, random splitting, or duplicate removal occurs.
    """

    def __init__(
        self,
        root: Path,
        split: Split,
        split_manifest: Path,
        box_policy: BoxPolicy = "strict",
    ) -> None:
        if box_policy not in ("strict", "clip"):
            raise ValueError(f"Unsupported box policy: {box_policy}")

        self.root = Path(root)
        self.split = split
        self.split_manifest = Path(split_manifest)
        self.box_policy = box_policy

        if split not in ("train", "val", "test"):
            raise ValueError(f"Unsupported split: {split}")

        self.annotations_path = self.root / "VG-SGG-with-attri.h5"
        self._annotations = ProcessLocalH5(self.annotations_path)
        self.metadata_path = self.root / "image_data.json"
        self.vocabulary_path = self.root / "VG-SGG-dicts-with-attri.json"

        with self.split_manifest.open(encoding="utf-8") as file:
            manifest = json.load(file)

        if not all(key in manifest for key in ("train", "val", "test")):
            raise ValueError("Manifest requires train, val, and test lists")

        for name in ("train", "val", "test"):
            indices = manifest[name]
            if not isinstance(indices, list) or any(type(index) is not int or index < 0 for index in indices):
                raise ValueError(f"Invalid indices in split: {name}")

            if len(indices) != len(set(indices)):
                raise ValueError(f"Duplicate indices in split: {name}")

        groups = [set(manifest[name]) for name in ("train", "val", "test")]
        if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("Train/val/test splits must be disjoint")

        self.indices: list[int] = manifest[split]

        with self.metadata_path.open(encoding="utf-8") as file:
            self.image_metadata = json.load(file)

        with self.vocabulary_path.open(encoding="utf-8") as file:
            self.vocabulary = json.load(file)

        if not isinstance(self.image_metadata, list):
            raise ValueError("Image metadata must be a list")

        for key in ("label_to_idx", "predicate_to_idx", "attribute_to_idx"):
            if key not in self.vocabulary:
                raise ValueError(f"Missing vocabulary: {key}")

        with h5py.File(self.annotations_path, "r") as h5:
            self.num_images = len(h5["split"])

            if len(self.image_metadata) != self.num_images:
                raise ValueError(
                    "Image metadata must align with HDF5 image rows. " "Build an aligned metadata file before loading."
                )

            for name in ("train", "val", "test"):
                if any(index >= self.num_images for index in manifest[name]):
                    raise ValueError(f"Out-of-range HDF5 index in {name}")

            expected_flags = {"train": 0, "val": 0, "test": 2}
            flags = h5["split"][:]

            for name in ("train", "val", "test"):
                for index in manifest[name]:
                    if int(flags[index]) != expected_flags[name]:
                        raise ValueError(f"Invalid HDF5 split flag for {name}: {index}")

    def __len__(self) -> int:
        """Return the number of selected images."""
        return len(self.indices)

    def load(self, index: int) -> SceneGraphRecord:
        """Return one normalized record using a split-relative index."""
        row = self.indices[index]

        with self._annotations.open() as h5:
            first_box = int(h5["img_to_first_box"][row])
            last_box = int(h5["img_to_last_box"][row])
            first_rel = int(h5["img_to_first_rel"][row])
            last_rel = int(h5["img_to_last_rel"][row])

            if first_box < 0 or last_box < first_box:
                raise ValueError(f"Image row {row} has invalid object range")

            info = self.image_metadata[row]
            width = int(info["width"])
            height = int(info["height"])
            image_id = str(info["image_id"])

            labels = h5["labels"][first_box : last_box + 1].reshape(-1)
            boxes = h5["boxes_1024"][first_box : last_box + 1]
            attributes = h5["attributes"][first_box : last_box + 1]

            scale = max(width, height) / BOX_SCALE
            objects = []

            for local_id, (label, box, attrs) in enumerate(zip(labels, boxes, attributes, strict=False)):
                cx, cy, bw, bh = (float(v) for v in box)

                xyxy = (
                    (cx - bw / 2) * scale,
                    (cy - bh / 2) * scale,
                    (cx + bw / 2) * scale,
                    (cy + bh / 2) * scale,
                )

                if self.box_policy == "clip":
                    converted_box = _clip_box_to_image(xyxy, width, height)
                else:
                    converted_box = _clip_rounding_overflow(xyxy, width, height, tolerance=0.5 * scale)

                objects.append(
                    SceneObject(
                        object_id=local_id,
                        label_id=int(label),
                        bbox_xyxy=converted_box,
                        attribute_ids=tuple(int(value) for value in attrs if int(value) > 0),
                    )
                )

            relations = []

            if first_rel >= 0:
                if last_rel < first_rel:
                    raise ValueError("Invalid relationship range")

                pairs = h5["relationships"][first_rel : last_rel + 1]
                predicates = h5["predicates"][first_rel : last_rel + 1].reshape(-1)

                for (subject, obj), predicate in zip(pairs, predicates, strict=False):
                    relations.append(
                        SceneRelation(
                            subject_id=int(subject) - first_box,
                            object_id=int(obj) - first_box,
                            predicate_id=int(predicate),
                        )
                    )
            elif last_rel != -1:
                raise ValueError("Inconsistent empty relationship range")

        record = SceneGraphRecord(
            image_id=image_id,
            width=width,
            height=height,
            objects=tuple(objects),
            relations=tuple(relations),
            metadata={
                "dataset": "vg150",
                "hdf5_row": str(row),
                "box_policy": self.box_policy,
            },
        )
        record.validate()
        return record


class VG150GraphDataset(Dataset):
    """PyG adapter; delegation will be connected in Milestone 4.3."""

    def __init__(
        self,
        loader: VG150AnnotationLoader,
        feature_extractor: FeatureExtractor,
        graph_builder: SceneGraphBuilder,
    ) -> None:
        super().__init__(root=None)
        self.loader = loader
        self.feature_extractor = feature_extractor
        self.graph_builder = graph_builder

    def len(self) -> int:
        """Return number of image graphs."""
        return len(self.loader)

    def get(self, idx: int) -> Data:
        """Build a PyG graph from an annotated image.

        The loader, feature extractor, and graph builder are independent
        components. Ground-truth relations are not used as node features
        or to construct the message-passing topology.
        """
        record = self.loader.load(idx)
        features = self.feature_extractor.extract(record)
        return self.graph_builder.build(record, features)
