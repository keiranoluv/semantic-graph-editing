"""Shared synthetic annotations (no VG150 assets)."""

import pytest

from src.datasets.schemas import SceneGraphRecord, SceneObject, SceneRelation


@pytest.fixture
def sample_scene() -> SceneGraphRecord:
    """Three objects and one directed, annotated predicate."""
    return SceneGraphRecord(
        image_id="synthetic_001",
        width=640,
        height=480,
        objects=(
            SceneObject(10, 1, (10.0, 20.0, 100.0, 200.0)),
            SceneObject(30, 2, (110.0, 70.0, 340.0, 300.0)),
            SceneObject(20, 3, (20.0, 300.0, 500.0, 470.0)),
        ),
        relations=(SceneRelation(10, 30, 1),),
    )
