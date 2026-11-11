"""Regression tests for our VG150 graph-constrained PredCls protocol."""

import pytest
import torch

from src.evaluation.scene_graph import evaluate_image, summarize_evaluation


def evaluate(logits, pairs, gt_pairs, predicates, ks=(1, 2)):
    return evaluate_image(
        logits=torch.tensor(logits, dtype=torch.float32),
        candidate_pairs=torch.tensor(pairs, dtype=torch.long),
        gt_pairs=torch.tensor(gt_pairs, dtype=torch.long),
        gt_predicates=torch.tensor(predicates, dtype=torch.long),
        ks=ks,
    )


def test_ranking_uses_scores_not_ground_truth_pairs():
    # A high-confidence incorrect pair outranks the correct one.
    result = evaluate(
        logits=[
            [0.0, 9.0, 0.0],
            [0.0, 3.0, 0.0],
        ],
        pairs=[[0, 1], [1, 2]],
        gt_pairs=[[1], [2]],
        predicates=[1],
    )

    assert result["hits"][1] == 0
    assert result["hits"][2] == 1


def test_background_never_becomes_predicate():
    # Background is strongest, but graph-constrained evaluation
    # still selects the best foreground predicate.
    result = evaluate(
        logits=[[10.0, 2.0, 0.0]],
        pairs=[[0], [1]],
        gt_pairs=[[0], [1]],
        predicates=[1],
    )

    assert result["hits"][1] == 1


def test_multiple_ground_truth_predicates():
    result = evaluate(
        logits=[[0.0, 9.0, 1.0]],
        pairs=[[0], [1]],
        gt_pairs=[[0, 0], [1, 1]],
        predicates=[1, 2],
    )

    assert result["support"] == 2
    assert result["hits"][2] == 1


def test_recall_is_image_macro():
    # Image A: 1/1 correct
    a = evaluate(
        logits=[[0.0, 9.0, 0.0]],
        pairs=[[0], [1]],
        gt_pairs=[[0], [1]],
        predicates=[1],
    )

    # Image B: 0/2 correct
    b = evaluate(
        logits=[[0.0, 0.0, 9.0]],
        pairs=[[0], [1]],
        gt_pairs=[[0, 0], [1, 2]],
        predicates=[1, 1],
    )

    summary = summarize_evaluation([a, b], ks=(1, 2))

    assert summary["R@2"] == pytest.approx(0.5)


def test_mean_recall_is_predicate_macro():
    # Predicate 1: 1 correct out of 1
    # Predicate 2: 0 correct out of 1
    result = evaluate(
        logits=[
            [0.0, 9.0, 0.0],
            [0.0, 9.0, 0.0],
        ],
        pairs=[[0, 1], [1, 0]],
        gt_pairs=[[0, 1], [1, 0]],
        predicates=[1, 2],
    )

    summary = summarize_evaluation([result], ks=(1, 2))

    assert summary["R@2"] == pytest.approx(0.5)
    assert summary["mR@2"] == pytest.approx(0.5)
    assert summary["num_predicates_with_gt"] == 2


def test_duplicate_ground_truth_is_deduplicated():
    result = evaluate(
        logits=[[0.0, 9.0]],
        pairs=[[0], [1]],
        gt_pairs=[[0, 0], [1, 1]],
        predicates=[1, 1],
    )

    assert result["support"] == 1
    assert result["hits"][1] == 1


def test_invalid_predicate_rejected():
    with pytest.raises(ValueError):
        evaluate(
            logits=[[0.0, 9.0]],
            pairs=[[0], [1]],
            gt_pairs=[[0], [1]],
            predicates=[2],
        )
