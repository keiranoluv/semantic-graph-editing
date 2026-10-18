"""Geometry-only extractor placeholder."""

import torch

from src.datasets.schemas import SceneGraphRecord
from src.graph.features import FeatureExtractor


class GeometryFeatureExtractor(FeatureExtractor):
    """Eventually encode normalized xyxy and related geometric quantities."""

    @property
    def output_dim(self) -> int:
        """Draft convention: normalized xyxy (4 components)."""
        return 4

    def extract(self, record: SceneGraphRecord) -> torch.Tensor:
        """Return [N, 4]; normalization is deferred to implementation."""
        raise NotImplementedError("Geometry encoding belongs to Phase 4")
