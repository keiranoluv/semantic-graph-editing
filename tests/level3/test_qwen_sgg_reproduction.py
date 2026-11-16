"""CPU fixtures for the two-command reproduction chain; no real image/SGG model runs."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
from PIL import Image

from tools.level3_image_graph_v1.core import read, sha256, write
from tools.level3_qwen21_v1.core import finish
from tools.level3_qwen_sgg import backend, core, runner
from tools.level3_qwen_unified import core as qwen


@pytest.fixture
def world(tmp_path, monkeypatch):
    cfg, qcfg = read(core.CONFIG), read(qwen.CONFIG)
    parent = core.ROOT / "configs/level3/sgg-pretrained-v2.json"
    saved_parent = tmp_path / "configs/level3/sgg-pretrained-v2.json"
    saved_parent.parent.mkdir(parents=True)
    saved_parent.write_bytes(parent.read_bytes())
    monkeypatch.setattr(core, "ROOT", tmp_path)
    monkeypatch.setattr(core, "RUNS", tmp_path / "runs/level3/qwen-sgg")
    monkeypatch.setattr(qwen, "ROOT", tmp_path)
    monkeypatch.setattr(qwen, "RUNS", tmp_path / "runs/level3/qwen200")
    tasks = []
    for i in range(200):
        source = tmp_path / f"source-{i % 100}.png"
        if not source.exists():
            Image.new("RGB", (100, 80), (i % 100, 0, 0)).save(source)
        tasks.append(
            dict(
                task_id=f"task-{i}",
                image_id=f"image-{i % 100}",
                source_path=str(source),
                sha256=sha256(source),
                dataset="synthetic-test-only",
                instruction_en=f"Add {i} cats.",
                operation={"kind": "add", "label": "cat"},
            )
        )
    manifest = tmp_path / "configs/level3/cohort.json"
    write(manifest, {"tasks": tasks})
    qcfg["manifest_sha256"] = cfg["manifest_sha256"] = sha256(manifest)
    cfg.update(
        manifest=str(manifest.relative_to(tmp_path)),
        pipeline_files_sha256={},
        qwen_run="runs/level3/qwen200/seed42",
        qwen_index_sha256="frozen-old-index",
    )
    config = tmp_path / "configs/level3/sgg-test.json"
    write(config, cfg)
    monkeypatch.setattr(core, "CONFIG", config)
    monkeypatch.setattr(core, "qwen_inputs", lambda: (qcfg, tasks))
    code = {"commit": "synthetic-CPU-fixture", "files_sha256": {}}
    directory = qwen.workspace(qwen.RUNS / "reproduce-01", qcfg, code)
    for i, task in enumerate(tasks):
        case = directory / "cases" / task["task_id"] / "attempt-0001"
        case.mkdir(parents=True)
        source, target = case / "source.png", case / "target.png"
        source.write_bytes(Path(task["source_path"]).read_bytes())
        Image.new("RGB", (100, 80), (128 + i % 128, 10 + i // 128, 30)).save(target)
        write(
            case / "result.json",
            dict(
                case_dir=str(case),
                task_id=task["task_id"],
                image_id=task["image_id"],
                dataset=task["dataset"],
                prompt=task["instruction_en"],
                operation=task["operation"],
                status="ok",
                native_source_sha256=task["sha256"],
                seed=42,
                dtype="bfloat16",
                steps_requested=40,
                completed_steps=40,
                source_path=str(source),
                source_sha256=sha256(source),
                target_path=str(target),
                target={"sha256": sha256(target)},
                human_quality_label=None,
            ),
        )
        finish(case)
    qwen.collect(directory, qcfg, tasks, [])
    return NS(cfg=cfg, qcfg=qcfg, directory=directory, tasks=tasks, code=code, root=tmp_path)


def test_selected_Qwen_run_supplies_all_300_images_and_preserves_frozen_config(world):
    w = world
    before = sha256(core.CONFIG)
    cfg, rows = core.inputs(w.directory.name)
    assert cfg["schema"] == "qwen-sgg-unified-v2"
    assert cfg["qwen_index_sha256"] == sha256(w.directory / "index.json")
    assert len(rows) == 200 and len(core.inventory(rows, cfg)) == 300
    assert all(w.directory in Path(r[role + "_path"]).parents for r in rows for role in ("source", "target"))
    assert core.protocol(cfg) == core.protocol(w.cfg)
    assert sha256(core.CONFIG) == before
    out = core.workspace(core.RUNS / "new", cfg, w.code)
    assert core.workspace(out, core.inputs(w.directory.name)[0], w.code) == out
    assert read(out / "run.json")["binding"]["qwen_run"] == cfg["qwen_run"]


@pytest.mark.parametrize("damage", ["partial", "protocol", "receipt", "image", "order", "binding"])
def test_reproduction_rejects_incomplete_or_changed_inputs(world, damage):
    w = world
    idx = read(w.directory / "index.json")
    case = Path(idx["entries"][0]["case_dir"])
    if damage == "partial":
        idx.update(successful=199, execution_complete=False)
        qwen.atomic(w.directory / "index.json", idx)
    elif damage == "protocol":
        header = read(w.directory / "run.json")
        header["protocol"]["generation"]["seed"] = 43
        qwen.atomic(w.directory / "run.json", header)
    elif damage == "receipt":
        qwen.atomic(case / "receipt.json", {"execution_complete": False})
    elif damage == "image":
        (case / "target.png").write_bytes(b"changed")
    else:
        if damage == "order":
            idx["entries"][0], idx["entries"][1] = idx["entries"][1], idx["entries"][0]
        else:
            idx["entries"][0]["binding"] = "different-image"
        qwen.atomic(w.directory / "index.json", idx)
    with pytest.raises(ValueError):
        core.inputs(w.directory.name)


def test_resume_cannot_switch_input_run_even_with_identical_images_and_index(world):
    w = world
    cfg, _ = core.inputs(w.directory.name)
    out = core.workspace(core.RUNS / "new", cfg, w.code)
    other = w.directory.with_name("reproduce-02")
    other.mkdir()
    for name in ("run.json", "index.json"):
        (other / name).write_bytes((w.directory / name).read_bytes())
    other_cfg, _ = core.inputs(other.name)
    assert other_cfg["qwen_index_sha256"] == cfg["qwen_index_sha256"]
    before = sha256(out / "run.json")
    with pytest.raises(ValueError, match="SGG_resume_protocol_changed"):
        core.workspace(out, other_cfg, w.code)
    assert sha256(out / "run.json") == before


def test_resume_rejects_changed_completed_Qwen_index_and_protects_old_SGG_output(world):
    w = world
    cfg, _ = core.inputs(w.directory.name)
    out = core.workspace(core.RUNS / "new", cfg, w.code)
    old = core.workspace(core.RUNS / "historical", w.cfg, w.code)
    with pytest.raises(ValueError, match="SGG_resume_protocol_changed"):
        core.workspace(old, cfg, w.code)
    idx = read(w.directory / "index.json")
    idx["warnings"].append("different-run-metadata")
    qwen.atomic(w.directory / "index.json", idx)
    with pytest.raises(ValueError, match="SGG_resume_protocol_changed"):
        core.workspace(out, core.inputs(w.directory.name)[0], w.code)


def test_default_input_keeps_original_frozen_index_guard(world):
    w = world
    cfg = copy.deepcopy(w.cfg)
    cfg.update(qwen_run=str(w.directory.relative_to(w.root)), qwen_index_sha256=sha256(w.directory / "index.json"))
    qwen.atomic(core.CONFIG, cfg)
    assert core.inputs()[0] == cfg
    idx = read(w.directory / "index.json")
    idx["warnings"].append("changed")
    qwen.atomic(w.directory / "index.json", idx)
    with pytest.raises(ValueError, match="frozen_200_image_pairs_changed"):
        core.inputs()


def test_input_path_cannot_escape_Qwen_output_root(world):
    with pytest.raises(ValueError, match="invalid_run_name"):
        core.inputs("../escape")


def test_graph_step_requires_the_matching_image_step_first(world):
    with pytest.raises(ValueError, match="run_step_1_first"):
        core.inputs("not-started")
    assert not (core.RUNS / "not-started").exists()


def test_cli_fresh_300_image_chain_interrupts_and_resumes_without_legacy_import(world, monkeypatch):
    w = world
    calls = []
    state = NS(fail_at=3, loads=0)

    class CPUFixture:
        def __init__(self, cfg, attempt):
            state.loads += 1
            assert cfg["qwen_run"] == str(w.directory.relative_to(w.root))

        def predict(self, path, raw):
            calls.append(path.parent.parent.name)
            if len(calls) == state.fail_at:
                raise RuntimeError("synthetic interruption; no GPU")
            np.savez_compressed(raw, synthetic_CPU_fixture=np.array([1]))
            return [], [], {"raw_npz": raw.name}, {"elapsed_seconds": 0, "gpu_memory": None}

        def close(self):
            return {"synthetic_CPU_fixture": True}

    tracking = NS(id="fixture", url="synthetic://no-network", log=lambda _: None, finish=lambda **_: None)
    infer = runner.infer_pending
    monkeypatch.setattr(runner, "identity", lambda: w.code)
    monkeypatch.setattr(backend, "preflight", lambda: {})
    monkeypatch.setattr(runner, "GPU_LOCK", w.root / "fixture.lock")
    monkeypatch.setattr(
        runner,
        "infer_pending",
        lambda *args: infer(*args, backend_factory=CPUFixture, tracker_factory=lambda **_: tracking),
    )
    monkeypatch.setattr(core, "legacy_predictions", lambda *args: pytest.fail("must not read old graphs"))
    out = core.RUNS / "reproduce-01"
    argv = ["sgg", "infer", "--output", str(out), "--max-new-images", "300"]
    monkeypatch.setattr(sys, "argv", argv)
    assert runner.main() == 2
    assert read(out / "index.json")["successful_images"] == 2
    state.fail_at = None
    assert runner.main() == 0
    assert len(calls) == 301 and len(set(calls)) == 300 and state.loads == 2
    assert read(out / "index.json")["successful_images"] == 300
    assert read(out / "review-data.json")["ready_pairs"] == 200
    monkeypatch.setattr(sys, "argv", argv[:-1] + ["0"])
    assert runner.main() == 0
    assert len(calls) == 301 and state.loads == 2


@pytest.mark.parametrize("run_id", ["seed42", "repeat-02"])
def test_two_launchers_share_one_run_name_and_do_not_import_old_runs(tmp_path, run_id):
    stub = tmp_path / "python-stub"
    stub.write_text("#!/usr/bin/env python3\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n")
    stub.chmod(0o755)
    env = {**os.environ, "PYTHON": str(stub), "RUN_ID": run_id}
    env.pop("OUTPUT", None)
    root = Path(__file__).resolve().parents[2]
    for step, folder in [("qwen", "qwen200"), ("sgg", "qwen-sgg")]:
        result = subprocess.run(
            ["bash", str(root / f"scripts/run_level3_{step}.sh")],
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        args = json.loads(result.stdout)
        assert args[args.index("--output") + 1] == f"runs/level3/{folder}/{run_id}"
        assert "infer" in args and "--no-import" in args
        assert "--qwen-run" not in args
        if step == "sgg":
            assert args[args.index("--max-new-images") + 1] == "300"
