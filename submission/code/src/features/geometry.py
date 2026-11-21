"""Bounding-box geometry features for scene graph nodes."""

import torch

from src.datasets.schemas import SceneGraphRecord
from src.graph.features import FeatureExtractor


class GeometryFeatureExtractor(FeatureExtractor):
    """Extract five normalized geometry features per object.

    Output columns:
        center_x, center_y, width, height, area

    All values are normalized using image dimensions.
    Object order is preserved.
    """

    @property
    def output_dim(self) -> int:
        """Return the fixed feature dimension."""
        return 5

    def extract(self, record: SceneGraphRecord) -> torch.Tensor:
        """Return geometry features with shape [N, 5]."""
        record.validate()

        features = []

        for obj in record.objects:
            x1, y1, x2, y2 = obj.bbox_xyxy

            width = (x2 - x1) / record.width
            height = (y2 - y1) / record.height
            center_x = (x1 + x2) / (2 * record.width)
            center_y = (y1 + y2) / (2 * record.height)
            area = width * height

            features.append([center_x, center_y, width, height, area])

        if not features:
            return torch.empty((0, self.output_dim), dtype=torch.float32)

        return torch.tensor(features, dtype=torch.float32)
