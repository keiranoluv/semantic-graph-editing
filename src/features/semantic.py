"""Label embedding extractor placeholder."""

import torch

from src.datasets.schemas import SceneGraphRecord
from src.graph.features import FeatureExtractor


class SemanticFeatureExtractor(FeatureExtractor):
    """Use frozen/precomputed object label embeddings; never learn here."""

    def __init__(self, embedding_dim: int) -> None:
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self._embedding_dim = embedding_dim

    @property
    def output_dim(self) -> int:
        """Embedding width configured by the eventual backend."""
        return self._embedding_dim

    def extract(self, record: SceneGraphRecord) -> torch.Tensor:
        """Return [N, D_label], preserving object order."""
        raise NotImplementedError("Frozen label embedding integration deferred")
