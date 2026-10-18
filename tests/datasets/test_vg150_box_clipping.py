"""Regression tests for minor VG150 bounding-box overflow."""

import pytest

from src.datasets.vg150 import _clip_rounding_overflow


def test_small_negative_x_overflow():
    """Correct a half-unit 1024-scale rounding overflow."""
    box = (-0.48828125, 13.0, 475.0, 336.0)

    result = _clip_rounding_overflow(
        box,
        width=662,
        height=1000,
        tolerance=0.5 * 1000 / 1024,
    )

    assert result == (0.0, 13.0, 475.0, 336.0)


def test_large_overflow_is_rejected():
    """Do not silently fix substantially invalid annotations."""
    with pytest.raises(ValueError, match="exceeds"):
        _clip_rounding_overflow(
            (-10.0, 10.0, 100.0, 100.0),
            width=662,
            height=1000,
            tolerance=0.5 * 1000 / 1024,
        )


def test_valid_box_unchanged():
    """Valid image-space coordinates are preserved."""
    box = (10.0, 20.0, 100.0, 200.0)

    assert (
        _clip_rounding_overflow(
            box,
            width=662,
            height=1000,
            tolerance=0.5 * 1000 / 1024,
        )
        == box
    )
