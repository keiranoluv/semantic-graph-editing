"""Frozen CLIP region vectors aligned to original VG150 HDF5 box rows."""

import json
from pathlib import Path

import numpy as np
import torch

from src.datasets.hdf5_session import ProcessLocalH5
from src.datasets.schemas import SceneGraphRecord
from src.features.geometry import GeometryFeatureExtractor
from src.graph.features import FeatureExtractor
from src.knowledge.transe.core import sha256


def load_visual_manifest(cache_dir: Path, root: Path, box_policy: str) -> dict:
    """Fail closed when a cache does not match the current annotations/split."""
    cache_dir = Path(cache_dir)
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("complete") is not True:
        raise ValueError("Visual cache is incomplete or uses an unsupported schema")
    if manifest.get("box_policy") != box_policy or manifest.get("visual_dim") != 512:
        raise ValueError("Visual cache box policy or feature width differs")
    for filename, digest in manifest["source_sha256"].items():
        if sha256(Path(root) / filename) != digest:
            raise ValueError(f"Visual cache source hash mismatch: {filename}")
    if not (cache_dir / "features.h5").is_file():
        raise FileNotFoundError(cache_dir / "features.h5")
    if sha256(cache_dir / "features.h5") != manifest.get("features_sha256"):
        raise ValueError("Visual cache feature file hash mismatch")
    return manifest


class VisualFeatureExtractor(FeatureExtractor):
    """Read precomputed object-region vectors in SceneGraphRecord order."""

    def __init__(self, cache_dir: Path, root: Path, box_policy: str = "clip") -> None:
        self.cache_dir = Path(cache_dir)
        self.manifest = load_visual_manifest(self.cache_dir, Path(root), box_policy)
        self.box_policy = box_policy
        self._features = ProcessLocalH5(self.cache_dir / "features.h5")

    @property
    def output_dim(self) -> int:
        return 512

    def extract(self, record: SceneGraphRecord) -> torch.Tensor:
        record.validate()
        if record.metadata.get("box_policy") != self.box_policy:
            raise ValueError("Record and visual cache box policies differ")
        row = int(record.metadata["hdf5_row"])
        with self._features.open() as cache:
            if not bool(cache["done"][row]):
                raise ValueError(f"Missing visual features for image row {row}")
            if int(cache["image_id"][row]) != int(record.image_id):
                raise ValueError(f"Visual cache image ID mismatch at row {row}")
            first, last = int(cache["row_first"][row]), int(cache["row_last"][row])
            if last - first + 1 != len(record.objects):
                raise ValueError(f"Visual cache object count mismatch at row {row}")
            boxes = cache["boxes_xyxy"][first : last + 1]
            expected = np.asarray([obj.bbox_xyxy for obj in record.objects], dtype=np.float32)
            if boxes.shape != expected.shape or not np.allclose(boxes, expected, atol=1e-3, rtol=0):
                raise ValueError(f"Visual cache GT boxes mismatch at row {row}")
            vectors = cache["features"][first : last + 1].astype(np.float32)
        if not np.isfinite(vectors).all():
            raise ValueError(f"Non-finite visual features at row {row}")
        return torch.from_numpy(vectors)


class GeometryVisualFeatureExtractor(FeatureExtractor):
    """Give all comparison models the same box geometry and region vectors."""

    def __init__(self, visual: VisualFeatureExtractor) -> None:
        self.geometry = GeometryFeatureExtractor()
        self.visual = visual

    @property
    def output_dim(self) -> int:
        return self.geometry.output_dim + self.visual.output_dim

    def extract(self, record: SceneGraphRecord) -> torch.Tensor:
        return torch.cat((self.geometry.extract(record), self.visual.extract(record)), dim=1)
