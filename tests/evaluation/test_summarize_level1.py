import copy

import pytest

from scripts.summarize_level1 import summarize


def reports():
    rows = []
    for seed, score in [(42, 0.1), (123, 0.2), (3407, 0.3)]:
        rows.append(
            {
                "model": "gat_kg",
                "seed": seed,
                "split": "val",
                "protocol": "PredCls_graph_constrained_top1_per_pair",
                "data_sha256": {"splits.json": "same"},
                "images_evaluated": 5000,
                "split_images": 5000,
                "full_split": True,
                "config": {
                    "model": "gat_kg",
                    "seed": seed,
                    "kg_manifest_sha256": "abc",
                    "use_knowledge": True,
                    "message_edges": "full",
                    "val_limit": 5000,
                },
                "per_predicate": {"1": {"support": 4}},
                "checkpoint_sha256": f"checkpoint-{seed}",
                "checkpoint": f"/tmp/{seed}.pt",
                "R@50": score,
                "R@100": score,
                "mR@50": score,
                "mR@100": score,
            }
        )
    return rows


def call(tmp_path, rows):
    paths = []
    for i, row in enumerate(rows):
        import json

        p = tmp_path / f"{i}.json"
        p.write_text(json.dumps(row))
        paths.append(p)
    return summarize(paths)


def test_three_seed_mean_and_sample_std(tmp_path):
    result = call(tmp_path, reports())
    assert result["groups"][0]["metrics"]["mR@50"]["mean"] == pytest.approx(0.2)
    assert result["groups"][0]["metrics"]["mR@50"]["sample_std"] == pytest.approx(0.1)


def test_conceptnet_group_has_correct_source_name(tmp_path):
    rows = reports()
    for row in rows:
        row["config"]["features"] = "geometry+frozen_clip_vit_b32+frozen_conceptnet+valid_bit"
    result = call(tmp_path, rows)
    assert result["groups"][0]["condition"].startswith("gat_kg:conceptnet:")


def test_pairwise_kg_factorial_summary_keeps_four_conditions(tmp_path):
    rows = []
    for kg_mode in ("off", "on"):
        for loss_mode in ("ce", "logit_adjusted"):
            for row in reports():
                row["model"] = "gat_pair_kg"
                row["config"].update(model="gat_pair_kg", kg_mode=kg_mode, loss_mode=loss_mode)
                row["checkpoint_sha256"] += f"-{kg_mode}-{loss_mode}"
                rows.append(row)
    result = call(tmp_path, rows)
    assert len(result["groups"]) == 4
    assert all(group["seeds"] == [42, 123, 3407] for group in result["groups"])


def test_clip_summary_rejects_cross_run_input_budget_drift(tmp_path):
    rows = reports()
    for row in rows:
        row["config"].update(
            visual_cache_manifest_sha256="same", geometry_dim=517, batch_size=512, code_sha256={"x": "a"}
        )
    rows[1]["config"]["batch_size"] = 256
    with pytest.raises(ValueError, match="shared input or training protocol"):
        call(tmp_path, rows)


@pytest.mark.parametrize(
    "change",
    ["subset", "different_split", "different_support", "different_config", "different_cache", "duplicate_seed"],
)
def test_rejects_noncomparable_reports(tmp_path, change):
    rows = copy.deepcopy(reports())
    if change == "subset":
        rows[1]["full_split"] = False
    elif change == "different_split":
        rows[1]["data_sha256"]["splits.json"] = "other"
    elif change == "different_support":
        rows[1]["per_predicate"]["1"]["support"] = 3
    elif change == "different_config":
        rows[1]["config"]["message_edges"] = "none"
    elif change == "different_cache":
        rows[1]["config"]["visual_cache_manifest_sha256"] = "other"
    else:
        rows[1]["seed"] = 42
        rows[1]["config"]["seed"] = 42
    with pytest.raises(ValueError):
        call(tmp_path, rows)
