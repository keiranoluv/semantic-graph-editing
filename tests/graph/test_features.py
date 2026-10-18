"""Verify feature contracts and deliberate Phase-4 placeholders."""

import pytest

from src.features.geometry import GeometryFeatureExtractor
from src.features.semantic import SemanticFeatureExtractor
from src.features.visual import VisualFeatureExtractor


def test_output_dimensions():
    assert GeometryFeatureExtractor().output_dim == 4
    assert SemanticFeatureExtractor(32).output_dim == 32
    assert VisualFeatureExtractor(64).output_dim == 64


def test_geometry_placeholder(sample_scene):
    with pytest.raises(NotImplementedError):
        GeometryFeatureExtractor().extract(sample_scene)
