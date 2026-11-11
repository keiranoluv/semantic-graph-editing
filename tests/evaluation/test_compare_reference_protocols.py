"""A protocol comparison must use identical predictions' source and GT support."""

import copy

import pytest

from tools.compare_reference_protocols import compare_reports


def test_comparison_rejects_changed_checkpoint_or_support():
    gc = {
        "checkpoint_sha256": "same",
        "checkpoint_epoch": 7,
        "seed": 42,
        "data_sha256": {"data": "same"},
        "config": {"box_policy": "same"},
        "split_images": 5000,
        "per_predicate": {"1": {"support": 9, "images_with_gt": 2, "R@50": 0.2}},
    }
    ng = copy.deepcopy(gc)
    ng["per_predicate"]["1"].pop("R@50")
    ng["per_predicate"]["1"]["ng-R@50"] = 0.4
    compare_reports(gc, ng)
    ng["checkpoint_sha256"] = "different"
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        compare_reports(gc, ng)
    ng["checkpoint_sha256"] = "same"
    ng["per_predicate"]["1"]["images_with_gt"] = 3
    with pytest.raises(ValueError, match="images_with_gt"):
        compare_reports(gc, ng)
