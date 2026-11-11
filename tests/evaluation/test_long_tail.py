"""Long-tail grouping must use only train and preserve reference class-macro recall."""

from collections import Counter

import h5py
import pytest

from src.evaluation.reference_predcls import summarize_reference
from tools.analyze_level1_long_tail import define_groups, grouped_recall, train_predicate_counts, write_fixed


def groups():
    return {"head": list(range(1, 11)), "body": list(range(11, 31)), "tail": list(range(31, 51))}


def test_frequency_counts_train_only_and_retains_duplicate_rows(tmp_path):
    with h5py.File(tmp_path / "data.h5", "w") as h5:
        h5["img_to_first_rel"] = [0, 3, 5, -1]
        h5["img_to_last_rel"] = [2, 4, 6, -1]
        h5["predicates"] = [[1], [1], [2], [50], [50], [49], [49]]
        counts = train_predicate_counts(h5, [0, 3])
        assert counts[1] == 2 and counts[2] == 1 and sum(counts.values()) == 3
        assert counts[49] == counts[50] == 0
        # A change confined to held-out labels cannot affect train grouping.
        h5["predicates"][3:] = [[3], [3], [3], [3]]
        assert train_predicate_counts(h5, [0, 3]) == counts
        with pytest.raises(ValueError, match="unique"):
            train_predicate_counts(h5, [0, 0])


def test_grouping_ties_use_id_and_partition_every_class():
    counts = {c: 10 for c in reversed(range(1, 51))}
    names = {c: str(c) for c in counts}
    definition = define_groups(counts, names)
    assert definition["groups"] == groups()
    assert definition["total_train_annotations"] == 500
    assert definition["group_train_support"]["tail"]["annotation_share"] == 0.4
    counts[50] = 100
    assert define_groups(counts, names)["groups"]["head"][0] == 50


def make_summary(prefix):
    rows = [
        {
            "support": 15,
            "hits": {50: 12, 100: 15},
            "class_support": Counter({1: 9, 2: 1, 31: 5}),
            "class_hits": {50: Counter({1: 9, 2: 1, 31: 2}), 100: Counter({1: 9, 2: 1, 31: 5})},
        },
        {
            "support": 1,
            "hits": {50: 0, 100: 0},
            "class_support": Counter({1: 1}),
            "class_hits": {50: Counter(), 100: Counter()},
        },
    ]
    return summarize_reference(rows, graph_constraint=not prefix)


@pytest.mark.parametrize("prefix", ["", "ng-"])
def test_group_means_keep_image_macro_and_absent_class_denominator(prefix):
    report = make_summary(prefix)
    summary = grouped_recall(report, groups(), prefix)
    # Class 1 = (9/9 + 0/1)/2 = .5, NOT 9/10. Class 2 = 1.
    assert summary["head"][f"{prefix}mR@50"] == pytest.approx(0.15)
    assert summary["tail"][f"{prefix}mR@50"] == pytest.approx(0.4 / 20)
    assert summary["tail"][f"{prefix}mR@100"] == pytest.approx(1 / 20)
    assert summary["body"][f"{prefix}mR@50"] == 0
    # Arithmetic mean over head/body/tail would be wrong because sizes differ.
    assert report[f"{prefix}mR@50"] == pytest.approx(0.2 * 0.15 + 0.4 * 0.02)


@pytest.mark.parametrize("error", ["missing_class", "absent_nonzero", "wrong_overall", "overlapping_groups"])
def test_invalid_report_or_partition_rejected(error):
    report, partition = make_summary(""), groups()
    if error == "missing_class":
        report["per_predicate"].pop("50")
    elif error == "absent_nonzero":
        report["per_predicate"]["50"]["R@50"] = 0.4
    elif error == "wrong_overall":
        report["mR@50"] = 0.9
    else:
        partition["tail"][-1] = 1
    with pytest.raises(ValueError):
        grouped_recall(report, partition, "")


def test_cannot_replace_frozen_group_definition(tmp_path):
    path = tmp_path / "groups.json"
    write_fixed(path, "original\n")
    write_fixed(path, "original\n")
    with pytest.raises(FileExistsError):
        write_fixed(path, "changed\n")
    assert path.read_text() == "original\n"
