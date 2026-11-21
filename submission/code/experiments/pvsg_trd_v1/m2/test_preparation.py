from copy import deepcopy

import numpy as np

from ..contracts import ROOT, read_json
from ..synthetic import synthetic_record, synthetic_video_targets
from .preparation import adapt_targets, internal_split, rasterize_adapted_targets, source_group


def test_grouped_split_preserves_outer_ids_and_never_uses_relations():
    manifest = read_json(ROOT / "manifests/pvsg_author_release_338_62.lock.json")
    result, audit = internal_split(manifest)
    assert result["outer_split"] == manifest["outer_split"]
    fit = result["internal_validation"]["fit_ids"]
    validation = result["internal_validation"]["validation_ids"]
    assert (len(fit), len(validation)) == (287, 51)
    assert not set(fit) & set(validation)
    assert set(fit) | set(validation) == set(manifest["outer_split"]["train_ids"])
    assert internal_split(manifest) == (result, audit)
    assert len(audit["official_train_test_shared_source_groups"]) == 4
    assert not (
        {source_group(v, result["video_sources"][v]) for v in fit}
        & {source_group(v, result["video_sources"][v]) for v in validation}
    )


def test_adaptation_records_every_filter_and_keeps_raw_unchanged():
    raw = synthetic_record()
    video, _ = synthetic_video_targets()
    original = deepcopy(raw)
    raw["relations"] += [
        [7, 12, "not_in_vocabulary", [[0, 9]]],
        [7, 999, raw["relations"][0][2], [[0, 9]]],
        [7, 12, raw["relations"][0][2], [[9, 100]]],
        [7, 12, raw["relations"][1][2], [[100, 110]]],
    ]
    before = deepcopy(raw)
    target, ledger, ignore = adapt_targets(raw, video)
    assert raw == before
    assert len(target["relations"]) == 2
    assert target["relations"][0]["intervals"] == [[2, 3], [7, 9]]
    assert {r["reason"] for r in ledger} == {
        "outside_frozen_ontology",
        "self_pair_or_unknown_track",
        "time_bounds",
        "no_covisible_support_in_mask_timeline",
    }
    assert len(ignore) == 2
    batch = rasterize_adapted_targets(video, target, ignore)
    index = {tuple(pair): i for i, pair in enumerate(batch["pairs"])}[7, 12]
    np.testing.assert_array_equal(batch["supervision_mask"][index, :, :2], batch["labels"][index, :, :2] > 0)
    assert batch["supervision_mask"][index, :, 2].sum() == 8
    assert adapt_targets(original, video)[1:] == ([], [])
