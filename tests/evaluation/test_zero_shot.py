from collections import Counter

import h5py
import pytest
import torch

from src.evaluation.zero_shot import evaluate_zero_shot_image, summarize_zero_shot, unseen_mask
from tools.level1_zero_shot import seen_from_rows


def test_seen_vocabulary_uses_train_classes_all_annotation_rows():
    with h5py.File("zero-shot-test", "w", driver="core", backing_store=False) as h5:
        h5["img_to_first_rel"] = [0, 3]
        h5["img_to_last_rel"] = [2, 3]
        h5["img_to_first_box"] = [0, 2]
        h5["img_to_last_box"] = [1, 3]
        h5["labels"] = [[4], [9], [4], [9]]
        h5["relationships"] = [[0, 1], [0, 1], [0, 1], [2, 3]]
        h5["predicates"] = [[1], [2], [2], [3]]
        seen, count = seen_from_rows(h5, [0])
        assert seen == {(4, 1, 9), (4, 2, 9)} and count == 3
        h5["predicates"][3] = [50]
        assert seen_from_rows(h5, [0]) == (seen, count)
        mask = unseen_mask(torch.tensor([[0, 0, 1], [1, 1, 0]]), torch.tensor([1, 3, 1]), torch.tensor([4, 9]), seen)
        assert mask.tolist() == [False, True, True]  # direction matters; node IDs are not class IDs


@pytest.mark.parametrize("constrained", [True, False])
def test_zero_shot_does_not_filter_predictions_before_topk(constrained):
    logits = torch.tensor([[0.0, 5.0, 1.0], [0.0, 1.0, 2.0]])
    pairs = torch.tensor([[0, 1], [2, 2]])
    boxes = torch.tensor([[0.0, 0.0, 5.0, 5.0], [10.0, 0.0, 15.0, 5.0], [20.0, 0.0, 25.0, 5.0]])
    result = evaluate_zero_shot_image(
        logits,
        pairs,
        pairs,
        torch.tensor([1, 2]),
        boxes,
        torch.tensor([1, 2, 3]),
        {(1, 1, 3)},
        ks=(1, 2),
        graph_constraint=constrained,
    )
    assert result["support"] == 1
    assert result["hits"] == {1: 0, 2: 1}


def test_multiple_predicates_same_pair_and_duplicate_zero_gt():
    pairs = torch.tensor([[0], [1]])
    args = (
        torch.tensor([[0.0, 5.0, 4.0]]),
        pairs,
        pairs.repeat(1, 3),
        torch.tensor([1, 2, 2]),
        torch.tensor([[0.0, 0.0, 5.0, 5.0], [10.0, 0.0, 15.0, 5.0]]),
        torch.tensor([4, 9]),
        {(4, 1, 9)},
    )
    gc = evaluate_zero_shot_image(*args, ks=(1, 2))
    ng = evaluate_zero_shot_image(*args, ks=(1, 2), graph_constraint=False)
    assert gc["support"] == ng["support"] == 2
    assert gc["hits"] == {1: 0, 2: 0}
    assert ng["hits"] == {1: 0, 2: 2}


def test_image_macro_not_pooled_and_empty_images_excluded():
    rows = [
        {"support": s, "hits": {50: h}, "class_support": Counter({1: s}) if s else Counter()}
        for s, h in [(9, 9), (1, 0), (0, 0)]
    ]
    summary = summarize_zero_shot(rows, ks=(50,))
    assert summary["zR@50"] == 0.5
    assert summary["images_with_zero_shot_gt"] == 2
    assert summary["zero_shot_gt_annotations"] == 10
    assert summarize_zero_shot(rows[2:], ks=(50,))["zR@50"] is None
    assert summarize_zero_shot([], graph_constraint=False)["ng-zR@50"] is None


def test_no_unseen_gt_has_zero_support_without_false_failure():
    pairs = torch.tensor([[0], [1]])
    result = evaluate_zero_shot_image(
        torch.tensor([[0.0, 1.0, 2.0]]),
        pairs,
        pairs,
        torch.tensor([2]),
        torch.tensor([[0.0, 0.0, 5.0, 5.0], [10.0, 0.0, 15.0, 5.0]]),
        torch.tensor([1, 2]),
        {(1, 2, 2)},
    )
    assert result["support"] == 0 and result["hits"] == {50: 0, 100: 0}
