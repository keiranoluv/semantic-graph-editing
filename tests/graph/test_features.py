"""Tests for scene graph feature extractors."""

import pytest
import torch

from src.features.geometry import GeometryFeatureExtractor
from src.features.semantic import SemanticFeatureExtractor


def test_output_dimensions():
    """Check configured output dimensions."""
    assert GeometryFeatureExtractor().output_dim == 5


def test_geometry_features(sample_scene):
    """Geometry extractor returns normalized features."""
    extractor = GeometryFeatureExtractor()
    features = extractor.extract(sample_scene)

    assert features.shape == (3, 5)
    assert features.dtype == torch.float32
    assert torch.isfinite(features).all()

    # Person bounding box:
    # (10, 20, 100, 200), image size: 640 x 480
    expected = torch.tensor(
        [
            55.0 / 640,
            110.0 / 480,
            90.0 / 640,
            180.0 / 480,
            (90.0 * 180.0) / (640 * 480),
        ],
        dtype=torch.float32,
    )

    torch.testing.assert_close(features[0], expected)


def test_semantic_placeholder(sample_scene):
    """Semantic features remain unimplemented."""
    with pytest.raises(NotImplementedError):
        SemanticFeatureExtractor(embedding_dim=300).extract(sample_scene)
