"""Tests for VG150 ordered candidate-pair statistics."""

import numpy as np
import pytest

from scripts.compute_vg150_statistics import count_pairs


def test_empty_graph_pairs():
    """Three objects produce six background candidates."""
    pairs = np.empty((0, 3), dtype=np.int64)

    assert count_pairs(pairs, 3) == (0, 0, 6)


def test_unique_positive_pairs():
    """Two distinct relations remove two background pairs."""
    pairs = np.array([[0, 1, 5], [1, 2, 7]])

    assert count_pairs(pairs, 3) == (2, 0, 4)


def test_multiple_predicates_on_same_pair():
    """One pair with two predicates is ambiguous."""
    pairs = np.array([[0, 1, 5], [0, 1, 7], [0, 1, 5]])

    assert count_pairs(pairs, 3) == (1, 1, 5)


def test_invalid_endpoint():
    """Reject invalid relationship endpoints."""
    with pytest.raises(ValueError, match="endpoint"):
        count_pairs(np.array([[0, 3, 5]]), 3)


def test_invalid_predicate():
    """Reject non-positive predicates."""
    with pytest.raises(ValueError, match="Predicate"):
        count_pairs(np.array([[0, 1, 0]]), 3)
