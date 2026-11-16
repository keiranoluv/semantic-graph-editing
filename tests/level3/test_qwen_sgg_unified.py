"""CPU fixtures verify per-image reuse, per-task binding, bounded execution and resume."""

import copy
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
from PIL import Image

from tools.level3_image_graph_v1.core import read, sha256, validate_record, write
from tools.level3_qwen21_v1.core import finish
from tools.level3_qwen_graphs_v1.core import normalize
from tools.level3_qwen_sgg import core, runner
from tools.level3_sgg_pretrained_v2.adapter import adapt


def decoded(empty=False):
    objects = (
        []
        if empty
        else [
            {"label": "person", "bbox_xyxy": [1, 2, 30, 60], "confidence": 0.9},
            {"label": "chair", "bbox_xyxy": [40, 2, 70, 60], "confidence": 0.8},
        ]
    )
    relations = (
        []
        if empty
        else [{"subject": 0, "object": 1, "predicate": "beside", "confidence": 0.7, "triplet_confidence": 0.6}]
    )
    return objects, relations


@pytest.fixture
def world(tmp_path, monkeypatch):
    cfg = read(core.CONFIG)
    cfg.update(legacy_run="old", legacy_config_sha256="old-config", pipeline_files_sha256={})
    monkeypatch.setattr(core, "ROOT", tmp_path)
    monkeypatch.setattr(core, "RUNS", tmp_path / "unified")
    rows = []
    for i in range(4):
        source = tmp_path / f"source-{i % 2}.png"
        if not source.exists():
            Image.new("RGB", (100, 80), (20 + i, 40, 50)).save(source)
        target = tmp_path / f"target-{i}.png"
        Image.new("RGBA", (100, 80), (80 + i, 90, 120, 255)).save(target)
        rows.append(
            dict(
                task_id=f"task-{i}",
                image_id=f"image-{i % 2}",
                dataset="test-only",
                prompt=f"Synthetic request {i}",
                source_path=str(source),
                source_sha256=sha256(source),
                target_path=str(target),
                target={"sha256": sha256(target)},
            )
        )
    old = tmp_path / "old"
    old.mkdir()
    write(old / "run.json", {"config": cfg, "config_sha256": "old-config", "code": {"files_sha256": {}}})
    pairs = []
    for row in rows[:2]:
        case = old / row["task_id"]
        case.mkdir()
        pair = {k: row[k] for k in ("task_id", "image_id", "dataset", "prompt")}
        pair["roles"] = {}
        for role in ("source", "target"):
            image = normalize(row[role + "_path"], case / (role + "-input.png"))
            image["image_id"] = row["task_id"] + "--" + role
            native = {"raw_npz": role + "-native.npz"}
            np.savez_compressed(case / native["raw_npz"], synthetic_cpu_fixture=np.array([1, 2]))
            objects, relations = decoded()
            write(case / (role + "-decoded.json"), {"objects": objects, "relations": relations, "native": native})
            rec = adapt(image, "react++", objects, relations, {}, native)
            rec.update(pair_task_id=row["task_id"], image_role=role)
            graph = case / (role + "-graph.json")
            write(graph, rec)
            pair["roles"][role] = {"status": "ok", "graph_path": str(graph)}
        pairs.append(pair)
    write(old / "results.json", pairs)
    finish(old)
    cfg["legacy_receipt_sha256"] = sha256(old / "receipt.json")
    code = {"commit": "CPU-synthetic-test-only", "files_sha256": {}}
    out = core.workspace(core.RUNS / "seed42", cfg, code)
    return NS(cfg=cfg, rows=rows, old=old, out=out, code=code)


def test_two_sources_four_targets_reuse_four_graphs_and_bind_four_pairs(world):
    w = world
    old_hashes = {p: sha256(p) for p in w.old.rglob("*") if p.is_file()}
    images = core.collect(w.out, w.cfg, w.rows)
    assert len(images) == 6 and sum(i["status"] == "ok" for i in images.values()) == 4
    pending = [u for i in images.values() if i["status"] != "ok" for u in i["uses"]]
    assert pending == [{"task_id": "task-2", "role": "target"}, {"task_id": "task-3", "role": "target"}]
    runner.refresh(w.out, w.cfg, w.rows, images)
    pairs = read(w.out / "results.json")
    a = read(pairs[0]["roles"]["source"]["graph_path"])
    b = read(pairs[2]["roles"]["source"]["graph_path"])
    assert a["source_graph"] == b["source_graph"] and a["grounding"] == b["grounding"]
    assert a["candidate_id"] == "task-0--source" and b["candidate_id"] == "task-2--source"
    assert a["pair_task_id"] != b["pair_task_id"]
    validate_record(a)
    validate_record(b)
    assert a["provenance"]["shared_image_prediction"] == b["provenance"]["shared_image_prediction"]
    assert a["source_graph"]["edges"][0]["subject"] == "n000"
    assert a["attribute_coverage"] == "unsupported_no_attribute_head" and a["human_reviewed"] is False
    assert Path(a["native_output"]["raw_npz_path"]).exists()
    assert {p: sha256(p) for p in old_hashes} == old_hashes
    assert read(w.out / "review-data.json")["ready_pairs"] == 2


@pytest.mark.parametrize("change", ["threshold", "seed", "normalization", "decoder", "image", "native", "receipt"])
def test_reuse_rejects_changed_model_image_or_native_evidence(world, change):
    w = world
    if change == "threshold":
        w.cfg["react"]["box_threshold"] = 0.9
    elif change == "seed":
        w.cfg["seed"] = 123
    elif change == "normalization":
        w.cfg["normalization"] = "unrelated_resize"
    elif change == "decoder":
        w.cfg["pipeline_files_sha256"] = {"changed_decoder.py": "different"}
    elif change == "image":
        w.rows[0]["target"]["sha256"] = "not-the-original-target"
    elif change == "native":
        next(w.old.rglob("*-native.npz")).write_bytes(b"changed")
    else:
        (w.old / "receipt.json").write_text("{}")
    with pytest.raises((ValueError, KeyError)):
        core.legacy_predictions(w.cfg, w.rows)


def test_resume_rejects_cached_artifact_tampering(world):
    w = world
    images = core.collect(w.out, w.cfg, w.rows)
    pred = next(i["prediction"] for i in images.values() if i["status"] == "ok")
    Path(pred["record_path"]).write_text("{}")
    with pytest.raises(ValueError, match="prediction_artifact_changed"):
        core.collect(w.out, w.cfg, w.rows, use_legacy=False)


def runtime():
    state = NS(calls=[], fail_at=None, bad_parse=False, empty=False, closed=0)

    class Backend:
        def __init__(self, cfg, attempt):
            assert cfg["seed"] == 42 and cfg["batch_size"] == 1

        def predict(self, path, raw):
            state.calls.append(str(path))
            np.savez_compressed(raw, synthetic_cpu_fixture=np.array([3]))
            if len(state.calls) == state.fail_at:
                raise RuntimeError("synthetic out of memory; no GPU was used")
            objects, relations = decoded(state.empty)
            if state.bad_parse:
                relations[0]["object"] = 999
            return objects, relations, {"raw_npz": raw.name}, {"elapsed_seconds": 0, "gpu_memory": None}

        def close(self):
            state.closed += 1
            return {"profile": {"synthetic_cpu_fixture": True}, "gpu_memory": None}

    tracking = NS(id="synthetic", url="synthetic://no-network", log=lambda _: None, finish=lambda **_: None)
    return state, Backend, lambda **_: tracking


def test_interruption_and_resume_infers_only_new_targets_keeps_evidence(world):
    w = world
    images = core.collect(w.out, w.cfg, w.rows)
    fake, backend, tracker = runtime()
    fake.fail_at = 2
    assert runner.infer_pending(w.out, w.cfg, w.rows, images, w.code, 2, backend, tracker) == 2
    assert read(w.out / "last-attempt.json")["error"]["oom"]
    assert read(w.out / "index.json")["successful_images"] == 5
    sealed = next(w.out.glob("predictions/*/attempt-*/receipt.json"))
    sealed_hash = sha256(sealed)
    images = core.collect(w.out, w.cfg, w.rows)
    fake.calls.clear()
    fake.fail_at = None
    assert runner.infer_pending(w.out, w.cfg, w.rows, images, w.code, 2, backend, tracker) == 0
    assert len(fake.calls) == 1 and sha256(sealed) == sealed_hash
    assert fake.closed == 2 and len(list(w.out.glob("predictions/*/attempt-*"))) == 3
    state = read(w.out / "index.json")
    assert state["execution_complete"] and state["successful_images"] == 6
    assert read(w.out / "review-data.json")["ready_pairs"] == 4
    assert all(i["prediction"]["origin"] == "legacy" for i in images.values() if len(i["uses"]) == 2)
    for pair in read(w.out / "results.json"):
        for role, item in pair["roles"].items():
            rec = read(item["graph_path"])
            assert rec["pair_task_id"] == pair["task_id"] and rec["image_role"] == role
            validate_record(rec)

    def never(*a, **kw):
        pytest.fail("complete cache must not load a model or contact W&B")

    assert runner.infer_pending(w.out, w.cfg, w.rows, images, w.code, 2, never, never) == 0


def test_budget_gate_before_model_and_tracking(world):
    w = world
    images = core.collect(w.out, w.cfg, w.rows, use_legacy=False)

    def never(*a, **kw):
        pytest.fail("budget check must happen before model/W&B")

    with pytest.raises(ValueError, match="budget_exceeded"):
        runner.infer_pending(w.out, w.cfg, w.rows, images, w.code, 2, never, never)


@pytest.mark.parametrize("empty", [False, True])
def test_parse_failure_is_not_an_empty_success(world, empty):
    w = world
    images = core.collect(w.out, w.cfg, w.rows)
    fake, backend, tracker = runtime()
    fake.empty, fake.bad_parse = empty, not empty
    status = runner.infer_pending(w.out, w.cfg, w.rows, images, w.code, 2, backend, tracker)
    assert status == (0 if empty else 2)
    assert len(list(w.out.glob("predictions/*/attempt-*/receipt.json"))) == (2 if empty else 0)
    assert read(w.out / "last-attempt.json")["quality_metrics"] is None


def test_output_protocol_gate_and_commit_gate(world, monkeypatch):
    w = world
    changed = copy.deepcopy(w.cfg)
    changed["react"]["relation_threshold"] = 0.99
    with pytest.raises(ValueError, match="resume_protocol_changed"):
        core.workspace(w.out, changed, w.code)
    monkeypatch.setattr(sys, "argv", ["sgg", "infer", "--output", str(w.out)])

    def dirty():
        raise ValueError("commit_SGG_code_before_execution")

    monkeypatch.setattr(runner, "identity", dirty)
    monkeypatch.setattr(runner, "inputs", lambda: pytest.fail("must stop at dirty code gate"))
    with pytest.raises(ValueError, match="commit_SGG"):
        runner.main()


def test_cached_pair_graphs_cannot_be_silently_overwritten(world):
    w = world
    images = core.collect(w.out, w.cfg, w.rows)
    runner.refresh(w.out, w.cfg, w.rows, images)
    path = w.out / "pairs/task-0/source-graph.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="refuse_to_overwrite"):
        runner.refresh(w.out, w.cfg, w.rows, images)
