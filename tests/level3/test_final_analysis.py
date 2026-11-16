import copy
import json

import numpy as np
import pytest
from scipy.stats import spearmanr

from tools.level3_analysis_v1 import core


def rows():
    return [
        {
            "task_id": f"t{i}",
            "image_id": f"source{i // 2}",
            "graph_consistency": i / 12,
            "clipscore": -i / 12,
            "negative_lpips": i / 6,
            "edit_score": i / 12,
            "evaluable_preservation": i / 12,
            "edit_fulfillment": i >= 6,
            "preservation": True,
            "overall_pass": i >= 6,
        }
        for i in range(12)
    ]


def test_spearman_ties_and_undefined_not_zero():
    x, y = [0, 0, 0.5, 1, 1], [0, 1, 0, 1, 1]
    assert core.rho(x, y) == pytest.approx(spearmanr(x, y).statistic)
    assert core.rho([1, 1], [False, True]) is None
    assert core.rho([0, 1], [True, True]) is None
    assert core.rho([], []) is None


def test_resample_keeps_both_edits_of_each_source_and_duplicates_clusters():
    data = rows()
    sampled = core.cluster_indices(data, np.random.default_rng(42))
    for a in range(0, len(data), 2):
        assert sampled.count(a) == sampled.count(a + 1)
    assert len(sampled) == len(data)
    assert any(sampled.count(i) > 1 for i in sampled)


def test_bootstrap_is_paired_deterministic_and_excludes_abstentions():
    data = rows()
    data.append({**data[0], "task_id": "missing", "image_id": "other", "graph_consistency": None})
    original = copy.deepcopy(data)
    a = core.bootstrap(data, replicates=100)
    assert a == core.bootstrap(data, replicates=100)
    assert data == original
    assert a["source_clusters"] == 6 and a["common_pairs"] == 12
    # An affine transform has identical ranks in every paired bootstrap sample.
    assert a["intervals"]["G_minus_negative_LPIPS"]["ci95"] == pytest.approx([0, 0])
    assert a["intervals"]["G_minus_CLIPScore"]["ci95"][0] > 0


def test_constant_label_bootstraps_remain_undefined():
    data = rows()[:4]
    b = core.bootstrap(data, replicates=20)
    assert all(x["ci95"] is None and x["undefined_replicates"] == 20 for x in b["intervals"].values())


@pytest.fixture
def notes(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "ROOT", tmp_path)
    data = rows()
    core.write(tmp_path / "results.json", data)
    core.write(tmp_path / "summary.json", {"preliminary": False, "all": core.comparison(data)})
    return {
        "schema": "level3-visual-review-v1",
        "human_ground_truth": False,
        "report": ".",
        "inputs_sha256": {n: core.sha(tmp_path / n) for n in ("results.json", "summary.json")},
        "cases": [
            {
                "task_id": r["task_id"],
                **{
                    f: "Visual observation, not a human label."
                    for f in ("visual_observation", "graph_evidence", "interpretation", "limitation")
                },
            }
            for r in core.disagreements(data)
        ],
    }


def test_analysis_refuses_changed_scores_or_labels(notes, tmp_path):
    core.validate_notes(notes)
    data = json.loads((tmp_path / "results.json").read_text())
    data[0]["overall_pass"] = True
    core.write(tmp_path / "results.json", data)
    with pytest.raises(ValueError, match="analysis_input_changed"):
        core.validate_notes(notes)


def test_analysis_refuses_cherry_picked_or_unreviewed_cases(notes):
    notes["cases"].reverse()
    with pytest.raises(ValueError, match="frozen_top10"):
        core.validate_notes(notes)
    notes["cases"].reverse()
    notes["cases"][0]["visual_observation"] = ""
    with pytest.raises(ValueError, match="missing_visual_review"):
        core.validate_notes(notes)


def test_media_binding_prevents_showing_wrong_image(tmp_path):
    source = tmp_path / "source.png"
    source.write_bytes(b"changed image")
    with pytest.raises(ValueError, match="image_or_graph_changed"):
        core.copy_media(source, tmp_path / "out", "wrong-sha")
