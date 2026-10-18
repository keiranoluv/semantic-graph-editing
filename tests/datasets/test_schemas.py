"""Schema tests exercise only behavior already implemented."""

import pytest

from src.datasets.schemas import SceneRelation


def test_valid_scene(sample_scene):
    assert sample_scene.image_id == "synthetic_001"
    sample_scene.validate()


def test_invalid_endpoint(sample_scene):
    from dataclasses import replace

    invalid = replace(sample_scene, relations=(SceneRelation(10, 999, 1),))
    with pytest.raises(ValueError, match="endpoint"):
        invalid.validate()
