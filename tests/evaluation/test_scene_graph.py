"""Synthetic tests for scene graph evaluation."""

import pytest
import torch

from src.evaluation.scene_graph import evaluate_image, summarize_evaluation


def test_perfect_predictions():
    logits = torch.tensor(
        [
            [0.0, 9.0, 0.0],
            [0.0, 0.0, 9.0],
        ]
    )
    pairs = torch.tensor([[0, 1], [1, 0]])
    predicates = torch.tensor([1, 2])

    result = evaluate_image(logits, pairs, pairs, predicates, ks=(1, 2))

    assert result["support"] == 2
    assert result["hits"][2] == 2

    summary = summarize_evaluation([result], ks=(1, 2))
    assert summary["R@2"] == pytest.approx(1.0)
    assert summary["mR@2"] == pytest.approx(1.0)


def test_multiple_gt_predicates_preserved():
    logits = torch.tensor([[0.0, 9.0, 1.0]])
    candidates = torch.tensor([[0], [1]])

    gt_pairs = torch.tensor([[0, 0], [1, 1]])
    gt_predicates = torch.tensor([1, 2])

    result = evaluate_image(logits, candidates, gt_pairs, gt_predicates, ks=(50,))

    assert result["support"] == 2
    assert result["hits"][50] == 1

    summary = summarize_evaluation([result], ks=(50,))
    assert summary["R@50"] == pytest.approx(0.5)
    assert summary["mR@50"] == pytest.approx(0.5)


def test_background_not_ranked():
    logits = torch.tensor([[10.0, 2.0, 0.0]])
    pairs = torch.tensor([[0], [1]])
    predicates = torch.tensor([1])

    result = evaluate_image(logits, pairs, pairs, predicates, ks=(50,))

    # Graph-constrained ranking forces a foreground prediction.
    assert result["hits"][50] == 1


def test_empty_ground_truth():
    logits = torch.tensor([[0.0, 1.0]])
    pairs = torch.tensor([[0], [1]])
    gt_pairs = torch.empty((2, 0), dtype=torch.long)
    gt_predicates = torch.empty(0, dtype=torch.long)

    result = evaluate_image(logits, pairs, gt_pairs, gt_predicates)
    summary = summarize_evaluation([result])

    assert result["support"] == 0
    assert summary["num_images_with_gt"] == 0
    assert summary["R@50"] is None


def test_invalid_shape():
    with pytest.raises(ValueError):
        evaluate_image(
            torch.zeros((2, 3)),
            torch.zeros((2, 1), dtype=torch.long),
            torch.zeros((2, 0), dtype=torch.long),
            torch.zeros(0, dtype=torch.long),
        )
