"""Synthetic tests for training pair sampling."""

import torch

from src.training.sampling import sample_training_pairs


def test_all_positive_pairs_preserved():
    labels = torch.tensor([1, 0, 2, 0, 3, 0, 0])
    indices = sample_training_pairs(labels)

    assert {0, 2, 4}.issubset(set(indices.tolist()))


def test_negative_sampling_ratio():
    labels = torch.tensor([1] + [0] * 100)
    indices = sample_training_pairs(labels, negative_ratio=3)

    selected = labels[indices]

    assert (selected > 0).sum().item() == 1
    assert (selected == 0).sum().item() == 3


def test_ambiguous_labels_ignored():
    labels = torch.tensor([1, -1, 0, 0, 2, -1])
    indices = sample_training_pairs(labels)

    assert not (labels[indices] == -1).any()


def test_reproducibility():
    labels = torch.tensor([1] + [0] * 30)

    generator_a = torch.Generator().manual_seed(42)
    generator_b = torch.Generator().manual_seed(42)

    a = sample_training_pairs(labels, generator=generator_a)
    b = sample_training_pairs(labels, generator=generator_b)

    torch.testing.assert_close(a, b)


def test_empty_positives():
    labels = torch.tensor([0, 0, -1])
    indices = sample_training_pairs(labels)

    assert indices.numel() == 0
