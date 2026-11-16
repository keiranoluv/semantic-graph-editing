import pytest
from PIL import Image

from tools.level3_image_graph_v1.core import digest, sha256, write
from tools.level3_qwen21_v1 import core
from tools.level3_shared.qwen_runtime import call_args, check_image


def test_output_immutable_and_cannot_touch_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "RUNS", tmp_path / "runs")
    path = core.output(tmp_path / "runs" / "v1")
    with pytest.raises(FileExistsError):
        core.output(path)
    with pytest.raises(ValueError):
        core.output(tmp_path / "data" / "locked")
    with pytest.raises(ValueError):
        core.output(tmp_path / "runs")


def test_receipt_rejects_modified_result(tmp_path):
    write(tmp_path / "results.json", [{"status": "ok"}])
    core.finish(tmp_path)
    assert core.bundle(tmp_path, "results.json")[0]["status"] == "ok"
    (tmp_path / "results.json").write_text("[]")
    with pytest.raises(ValueError, match="changed_artifact"):
        core.bundle(tmp_path, "results.json")


def test_receipt_rejects_failed_execution(tmp_path):
    write(tmp_path / "results.json", [])
    core.finish(tmp_path, success=False)
    with pytest.raises(ValueError, match="incomplete"):
        core.bundle(tmp_path, "results.json")


def test_pilot_needs_real_matching_smoke(tmp_path, monkeypatch):
    cfg = {"smoke_task": "chair", "pilot_tasks": ["a", "b", "c", "d", "e"]}
    index = {k: {"task_id": k} for k in ["chair", *cfg["pilot_tasks"]]}
    conf = tmp_path / "cfg.json"
    lock = tmp_path / "lock.json"
    write(conf, {})
    write(lock, {})
    monkeypatch.setattr(core, "CONFIG", conf)
    monkeypatch.setattr(core, "LOCK", lock)
    run = tmp_path / "smoke"
    run.mkdir()
    write(
        run / "run.json",
        {"mode": "smoke", "config_sha256": sha256(conf), "asset_lock_sha256": sha256(lock), "device": "cuda"},
    )
    write(run / "results.json", [{"task_id": "chair", "status": "ok"}])
    core.finish(run)
    assert [r["task_id"] for r in core.select(cfg, index, "pilot", run)] == cfg["pilot_tasks"]
    assert len(core.select(cfg, index, "smoke")) == 1
    with pytest.raises(ValueError, match="successful_smoke"):
        core.select(cfg, index, "pilot")
    conf.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="matching_GPU"):
        core.select(cfg, index, "pilot", run)


def test_smoke_cannot_be_cpu_or_wrong_task(tmp_path, monkeypatch):
    cfg = {"smoke_task": "chair", "pilot_tasks": ["a"]}
    conf = tmp_path / "cfg.json"
    lock = tmp_path / "lock.json"
    write(conf, {})
    write(lock, {})
    monkeypatch.setattr(core, "CONFIG", conf)
    monkeypatch.setattr(core, "LOCK", lock)
    run = tmp_path / "smoke"
    run.mkdir()
    write(
        run / "run.json",
        {"mode": "smoke", "config_sha256": sha256(conf), "asset_lock_sha256": sha256(lock), "device": "cpu"},
    )
    write(run / "results.json", [{"task_id": "chair", "status": "ok"}])
    core.finish(run)
    with pytest.raises(ValueError, match="matching_GPU"):
        core.select(cfg, {}, "pilot", run)


def test_edit_call_keeps_input_image_and_forty_steps():
    im = Image.new("RGB", (512, 288))
    generator = object()
    args = call_args(
        {"prompt": "Remove yellow chair.", "output_resolution": 1024, "true_cfg_scale": 1.0, "use_kv_cache": True},
        im,
        generator,
    )
    assert args["image"] is im and args["generator"] is generator
    assert args["num_inference_steps"] == 40 and args["num_images_per_prompt"] == 1
    assert args["output_resolution"] == 1024


def test_blank_output_is_technical_failure_not_pass(tmp_path):
    p = tmp_path / "black.png"
    Image.new("RGB", (96, 64)).save(p)
    with pytest.raises(ValueError, match="degenerate"):
        check_image(p)
    im = Image.new("RGB", (96, 64))
    im.putpixel((10, 10), (255, 255, 255))
    im.save(p)
    assert check_image(p)["sha256"] == sha256(p)


def test_manual_labels_start_unrated_and_bound_to_results():
    rows = [{"task_id": "chair", "status": "ok"}]
    template = core.review_template(rows)
    assert template["results_sha256"] == digest(rows)
    assert len(template["reviews"]) == 2
    for row in template["reviews"]:
        assert row["status"] == "pending" and row["edit_fulfillment"] is None and row["preservation"] is None
