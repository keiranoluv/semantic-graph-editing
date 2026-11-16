import copy
from pathlib import Path

import pytest

from tools.level3_evaluation import core, metrics, report, scoring
from tools.level3_sgg_pretrained_v2.adapter import adapt

CFG = {"matching_iou": 0.3, "ambiguity_margin": 0.05, "min_edit_object_confidence": 0.4, "relabel_iou": 0.5}


def graph(edges=(), duplicate=False):
    objects = [
        {"label": "chair", "bbox_xyxy": [10, 10, 30, 40], "confidence": 0.9},
        {"label": "person", "bbox_xyxy": [60, 20, 80, 90], "confidence": 0.9},
    ]
    if duplicate:
        objects.append(dict(objects[0]))
    return adapt(
        {"image_id": "fixture", "sha256": "synthetic", "width": 100, "height": 100},
        "react++",
        objects,
        [{"subject": a, "predicate": p, "object": b, "confidence": 0.8} for a, p, b in edges],
        {},
        {},
    )


def record(i=0, kind="attribute", target=None):
    return {
        "task": {
            "task_id": f"t{i}",
            "image_id": f"image{i//2}",
            "dataset": "fixture",
            "operation": {"kind": kind, "label": "chair"},
            "instruction_en": "Change the chair to blue.",
        },
        "author": {
            "grounding": {"source_node": "n000", "region": [0.1, 0.1, 0.3, 0.4], "binding_status": "bound"},
            "target_caption_en": "A blue chair.",
        },
        "binding": f"binding{i}",
        "source": graph([(0, "beside", 1)]),
        "target": target if target is not None else graph([(0, "beside", 1)]),
    }


def review(records):
    return {
        "schema": "qwen-unified-review-v1",
        "cohort_sha256": "cohort",
        "reviews": [
            {
                "task_id": r["task"]["task_id"],
                "binding": r["binding"],
                "status": "reviewed",
                "edit_fulfillment": i % 2 == 0,
                "preservation": True,
            }
            for i, r in enumerate(records)
        ],
    }


def test_color_cannot_pass_even_for_identical_graphs():
    r = record()
    original = copy.deepcopy(r)
    s = scoring.score(r, CFG)
    assert s["edit_score"] is s["graph_consistency"] is None
    assert s["preservation_score"] == 1 and s["preservation_evaluable"]
    assert s["reason_codes"] == ["unsupported_color_attribute"]
    assert r == original


def test_color_still_checks_directed_relation_preservation():
    s = scoring.score(record(target=graph([(1, "beside", 0)])), CFG)
    assert s["preservation_score"] == 0.5
    assert s["graph_consistency"] is None


def test_ambiguous_alignment_is_not_valid_component_evidence():
    s = scoring.score(record(target=graph([(0, "beside", 1)], duplicate=True)), CFG)
    assert not s["preservation_evaluable"]
    assert "ambiguous_preservation_alignment" in s["reason_codes"]


def test_inherited_remove_rule_and_scores_unchanged():
    from tools.level3_qwen_scoring_v1.scoring import score

    r = record(kind="remove")
    before = score(r["source"], r["target"], r["task"], r["author"]["grounding"], CFG)
    after = scoring.score(r, CFG)
    assert {k: after[k] for k in before} == before
    assert after["graph_consistency"] == 0


@pytest.mark.parametrize(
    "problem", ["duplicate", "unknown", "binding", "numeric_label", "missing_field", "wrong_cohort", "status"]
)
def test_reject_bad_human_review(problem):
    records = [record(0), record(1)]
    pack = review(records)
    if problem == "duplicate":
        pack["reviews"].append(pack["reviews"][0])
    if problem == "unknown":
        pack["reviews"][0]["task_id"] = "foreign"
    if problem == "binding":
        pack["reviews"][0]["binding"] = "new-image"
    if problem == "numeric_label":
        pack["reviews"][0]["edit_fulfillment"] = 1
    if problem == "missing_field":
        del pack["reviews"][0]["preservation"]
    if problem == "wrong_cohort":
        pack["cohort_sha256"] = "other"
    if problem == "status":
        pack["reviews"][0]["status"] = "pending"
    with pytest.raises(ValueError):
        report.validate_reviews(pack, records, "cohort")


def test_pending_labels_are_not_failures_and_partial_is_explicit():
    records = [record(0), record(1)]
    pack = review(records)
    pack["reviews"][1].update(edit_fulfillment=None, status="pending")
    with pytest.raises(ValueError, match="need_all"):
        report.validate_reviews(pack, records, "cohort")
    labels = report.validate_reviews(pack, records, "cohort", allow_partial=True)
    assert list(labels) == ["t0"] and labels["t0"]["overall_pass"] is True


def test_spearman_ties_constants_and_common_case_comparison():
    rows = [
        {
            "task_id": f"t{i}",
            "graph_consistency": v,
            "clipscore": v * 2,
            "negative_lpips": v - 1,
            "edit_score": v,
            "evaluable_preservation": v,
            "overall_pass": bool(v),
            "edit_fulfillment": bool(v),
            "preservation": bool(v),
        }
        for i, v in enumerate([0.0, 0.0, 1.0, 1.0])
    ]
    rows.append({**rows[0], "task_id": "missing", "graph_consistency": None})
    result = report.comparison(rows)
    assert result["common_n"] == 4
    for x in result["common_case_correlations"].values():
        assert x["n"] == 4 and x["spearman"] == 1
        assert x["task_ids"] == ["t0", "t1", "t2", "t3"]
    assert result["all_available_secondary_correlations"]["clipscore"]["n"] == 5
    constant = [{**r, "overall_pass": True} for r in rows]
    assert report.correlation(constant, "clipscore", "overall_pass")["spearman"] is None
    assert report.correlation([], "clipscore", "overall_pass")["reason"] == "fewer_than_two_cases"


def test_disagreement_candidates_are_ranked_but_not_auto_analysed():
    rows = [
        {
            "task_id": f"t{i}",
            "graph_consistency": i,
            "clipscore": 4 - i,
            "negative_lpips": i,
            "overall_pass": bool(i % 2),
        }
        for i in range(5)
    ]
    result = report.disagreements(rows, 2)
    assert [r["task_id"] for r in result] == ["t0", "t4"]
    assert all(r["qualitative_analysis"] is None for r in result)


def test_smoke_and_wrong_pair_stages_cannot_enter_report():
    records = [record()]
    pack = {"evaluation_eligible": False, "stage_binding": "s", "rows": [core.case_header(records[0])]}
    with pytest.raises(ValueError, match="smoke"):
        report.validate_stage(pack, records, "s")
    pack["evaluation_eligible"] = True
    pack["rows"][0]["binding"] = "wrong"
    with pytest.raises(ValueError, match="binding"):
        report.validate_stage(pack, records, "s")


def test_receipt_detects_tampering(tmp_path):
    p = tmp_path / "results.json"
    core.save_stage(p, {"rows": []})
    assert core.read_stage(p) == {"rows": []}
    p.write_text('{"rows":[1]}')
    with pytest.raises(ValueError, match="changed"):
        core.read_stage(p)


def test_metric_resume_binding_and_device():
    header = core.case_header(record())
    pack = {"case": header, "stage_binding": "s", "status": "ok", "device": "cpu", "values": {"clipscore": 0.3}}
    metrics.validate_cache(pack, header, "s", "cpu")
    with pytest.raises(ValueError):
        metrics.validate_cache(pack, header, "changed", "cpu")
    with pytest.raises(ValueError):
        metrics.validate_cache(pack, header, "s", "cuda")


def test_metrics_resume_keeps_completed_cases_after_interruption(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    records = [record(0), record(1)]
    calls = []
    fail = [True]

    class Fake:
        def __init__(self, device):
            pass

        def score(self, r):
            calls.append(r["task"]["task_id"])
            if r["task"]["task_id"] == "t1" and fail[0]:
                fail[0] = False
                raise RuntimeError("synthetic interruption")
            return {"clipscore": 0.5, "lpips": 0.2, "negative_lpips": -0.2}

    monkeypatch.setattr(metrics, "check_assets", lambda: {})
    monkeypatch.setattr(metrics, "check_captions", lambda r: {})
    monkeypatch.setattr(metrics, "ClipScore", Fake)
    monkeypatch.setattr(metrics, "LPIPS", Fake)
    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(
            init=lambda **kw: SimpleNamespace(id="fake", url="fake", log=lambda x: None, finish=lambda **kw: None)
        ),
    )
    cfg = {"seed": 42}
    with pytest.raises(RuntimeError, match="interruption"):
        metrics.run(tmp_path, cfg, records, "cpu")
    first = tmp_path / "metrics/clip/t0.json"
    checksum = core.sha256(first)
    metrics.run(tmp_path, cfg, records, "cpu")
    assert core.sha256(first) == checksum
    assert calls.count("t0") == 2  # once CLIP + once LPIPS; CLIP was resumed
    pack = core.read_stage(tmp_path / "metrics/results.json")
    assert len(pack["rows"]) == 2 and pack["evaluation_eligible"]


def test_prepared_metadata_covers_exactly_200_without_human_labels():
    cfg = core.read(core.CONFIG)
    tasks = core.read(core.ROOT / cfg["cohort"])["tasks"]
    authored = core.read(core.ROOT / cfg["authoring"])["tasks"]
    assert len(authored) == 200
    assert [r["task_id"] for r in authored] == [t["task_id"] for t in tasks]
    assert all(r["author"] == "assistant" and r["human_ground_truth"] is False for r in authored)
    assert all("target_caption_en" in r and "grounding" in r for r in authored)
    assert sum(t["operation"]["kind"] == "attribute" for t in tasks) == 10
    assert all(not any(k in r for k in report.FIELDS) for r in authored)


def test_complete_report_join_exports_and_preserves_previous_report(tmp_path, monkeypatch):
    import json

    from PIL import Image

    monkeypatch.setattr(report, "ROOT", tmp_path)
    core.atomic(tmp_path / "cohort.json", {"fixture": True})
    cfg = {"cohort": "cohort.json", "evaluation_protocol": {"fixture": True}}
    records = [record(i, kind="remove") for i in range(4)]
    image = tmp_path / "image.png"
    Image.new("RGB", (48, 48), "blue").save(image)
    for r in records:
        r["image"] = {"source_path": str(image), "target_path": str(image)}
    pack = review(records)
    pack["cohort_sha256"] = core.sha256(tmp_path / "cohort.json")
    core.atomic(tmp_path / "labels.json", pack)
    binding = core.stage_binding(cfg, records)
    graph_rows, metric_rows = [], []
    for i, r in enumerate(records):
        v = float(i % 2 == 0)
        graph_rows.append(
            {
                **core.case_header(r),
                "scores": {
                    "edit_score": v,
                    "preservation_score": 1.0,
                    "preservation_evaluable": True,
                    "graph_consistency": v,
                    "reason_codes": [],
                },
            }
        )
        metric_rows.append({**core.case_header(r), "clipscore": 1 - v, "lpips": 1 - v, "negative_lpips": v - 1})
    for kind, rows in [("graph", graph_rows), ("metrics", metric_rows)]:
        core.save_stage(
            tmp_path / kind / "results.json", {"stage_binding": binding, "evaluation_eligible": True, "rows": rows}
        )
    result = report.run(tmp_path, cfg, records, tmp_path / "labels.json")
    assert result["labeled"] == result["common_cases"] == 4 and not result["preliminary"]
    dest = Path(result["output"])
    summary = json.loads((dest / "summary.json").read_text())
    assert summary["all"]["common_case_correlations"]["graph_consistency"]["spearman"] == 1
    assert summary["all"]["common_case_correlations"]["clipscore"]["spearman"] == -1
    assert (dest / "comparison.html").is_file() and (dest / "results.csv").is_file()
    with pytest.raises(ValueError, match="already_exists"):
        report.run(tmp_path, cfg, records, tmp_path / "labels.json")
