"""Tests for explicit VG150 bounding-box preprocessing policies."""

import pytest

from src.datasets.vg150 import _clip_box_to_image


def test_major_overflow_can_be_clipped():
    """Out-of-bound coordinates are clipped when requested."""
    box = (-10.0, 20.0, 700.0, 300.0)

    result = _clip_box_to_image(box, width=662, height=1000)

    assert result == (0.0, 20.0, 662.0, 300.0)


def test_valid_box_is_unchanged():
    """Clipping must not alter an already valid box."""
    box = (10.0, 20.0, 100.0, 200.0)

    assert _clip_box_to_image(box, 662, 1000) == box


def test_degenerate_box_rejected():
    """Reject boxes that lose their area after clipping."""
    with pytest.raises(ValueError, match="Degenerate"):
        _clip_box_to_image(
            (-100.0, 20.0, -10.0, 100.0),
            width=662,
            height=1000,
        )


def test_nonfinite_box_rejected():
    """Reject NaN or infinity coordinates."""
    with pytest.raises(ValueError, match="finite"):
        _clip_box_to_image(
            (float("nan"), 10.0, 100.0, 200.0),
            width=662,
            height=1000,
        )


def test_invalid_source_geometry_rejected():
    """Reject boxes with inverted source coordinates."""
    with pytest.raises(ValueError, match="geometry"):
        _clip_box_to_image(
            (100.0, 20.0, 10.0, 200.0),
            width=662,
            height=1000,
        )
