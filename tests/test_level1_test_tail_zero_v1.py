import copy

import pytest

from src.evaluation.zero_shot import summarize_zero_shot
from tools.level1_test_tail_zero_v1 import summarize_images, verify_report


def test_subset_inference_same_macro_as_full_split():
    # Omitting images without unseen GT changes neither denominator nor ranking.
    full = [
        {"support": 9, "hits": {50: 9, 100: 9}, "class_support": {1: 9}},
        {"support": 0, "hits": {50: 0, 100: 0}, "class_support": {}},
        {"support": 1, "hits": {50: 0, 100: 1}, "class_support": {2: 1}},
    ]
    compact = [[0, 9, 9, 9, 9, 9], [2, 1, 0, 1, 0, 1]]
    result = summarize_images(compact)
    expected = summarize_zero_shot(full)
    assert result["zR@50"] == expected["zR@50"] == 0.5  # not pooled 9/10
    assert result["zR@100"] == expected["zR@100"] == 1


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [[0, 0, 0, 0, 0, 0]],
        [[0, 1, 2, 2, 1, 1]],
        [[0, 2, 2, 1, 1, 1]],
        [[0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 1]],
    ],
)
def test_reject_corrupt_per_image_rows(rows):
    with pytest.raises(ValueError):
        summarize_images(rows)


def fixture_report():
    from src.evaluation.zero_shot import PROTOCOL

    entry = {"condition": "fusion_baseline", "seed": 42, "checkpoint_sha256": "checkpoint"}
    freeze = {"cohort_sha256": "cohort", "source_sha256": {"source": "hash"}}
    cohort = {"images": [[8, 103, 9, 2]]}
    row = {
        "protocol": PROTOCOL,
        "split": "test",
        **entry,
        **freeze,
        "split_images": 26446,
        "images_inferred": 1,
        "images": [[8, 2, 1, 2, 1, 2]],
        "metrics": {"zR@50": 0.5, "zR@100": 1.0, "ng-zR@50": 0.5, "ng-zR@100": 1.0},
    }
    return row, entry, freeze, cohort


@pytest.mark.parametrize(
    "field,value",
    [
        ("split", "val"),
        ("checkpoint_sha256", "changed"),
        ("source_sha256", {}),
        ("cohort_sha256", "changed"),
        ("images", [[9, 2, 1, 2, 1, 2]]),
        ("metrics", {"zR@50": 1.0, "zR@100": 1.0, "ng-zR@50": 0.5, "ng-zR@100": 1.0}),
    ],
)
def test_resume_rejects_tampering(field, value):
    row, entry, freeze, cohort = fixture_report()
    verify_report(row, entry, freeze, cohort)
    changed = copy.deepcopy(row)
    changed[field] = value
    with pytest.raises(ValueError):
        verify_report(changed, entry, freeze, cohort)
