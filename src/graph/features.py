"""Common feature extraction contract."""

from abc import ABC, abstractmethod

import torch

from src.datasets.schemas import SceneGraphRecord


class FeatureExtractor(ABC):
    """Produce float32 features with one row per ordered object instance.

    All extractors must preserve SceneGraphRecord.objects order.
    """

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Feature width D, fixed for this extractor configuration."""

    @abstractmethod
    def extract(self, record: SceneGraphRecord) -> torch.Tensor:
        """Return tensor of shape [num_objects, output_dim]."""
