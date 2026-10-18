"""Region embedding extractor placeholder (no detector training)."""

import torch

from src.datasets.schemas import SceneGraphRecord
from src.graph.features import FeatureExtractor


class VisualFeatureExtractor(FeatureExtractor):
    """Eventually load precomputed or frozen-backbone region features."""

    def __init__(self, embedding_dim: int) -> None:
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self._embedding_dim = embedding_dim

    @property
    def output_dim(self) -> int:
        """Configured frozen region-encoder embedding width."""
        return self._embedding_dim

    def extract(self, record: SceneGraphRecord) -> torch.Tensor:
        """Return [N, D_visual], preserving object order."""
        raise NotImplementedError("Precomputed region feature integration deferred")
