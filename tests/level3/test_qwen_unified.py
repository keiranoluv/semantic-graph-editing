"""CPU evidence for 200-request import/resume, review bindings and lossless disk cleanup.

Synthetic editor outputs are test fixtures, never benchmark/model-quality results.
"""

import copy
import os
import sys
from types import SimpleNamespace as NS

import pytest
from PIL import Image

from tools.level3_image_graph_v1.core import read, sha256, write
from tools.level3_qwen21_v1.core import finish
from tools.level3_qwen_unified import cleanup, core, runner, viewer


@pytest.fixture
def world(tmp_path, monkeypatch):
    cfg = read(core.CONFIG)
    monkeypatch.setattr(core, "ROOT", tmp_path)
    monkeypatch.setattr(core, "RUNS", tmp_path / "runs/level3/qwen200")
    monkeypatch.setattr(cleanup, "ROOT", tmp_path)
    source = tmp_path / "original.png"
    Image.new("RGB", (32, 24), "green").save(source)
    tasks = [
        dict(
            task_id=f"task-{i}",
            image_id=f"image-{i % 2}",
            source_path=str(source),
            sha256=sha256(source),
            dataset="sidewalk",
            instruction_en=f"Add {i} cats.",
            visual_evidence="Synthetic source",
            operation={"kind": "add", "label": "cat"},
        )
        for i in range(4)
    ]
    code = {"commit": "synthetic-test-only", "files_sha256": {}}
    out = core.workspace(core.RUNS / "seed42", cfg, code)

    def legacy(name):
        path = tmp_path / "runs/level3" / name
        path.mkdir()
        write(
            path / "run.json",
            {
                "manifest_sha256": cfg["accepted_legacy_manifest_sha256"][0],
                "config": cfg,
                "asset_lock_sha256": cfg["asset_lock_sha256"],
            },
        )
        return path

    def case(parent, task, color="yellow", sealed=True):
        path = parent / "cases" / task["task_id"]
        path.mkdir(parents=True)
        s, t = path / "source.png", path / "target.png"
        s.write_bytes(source.read_bytes())
        Image.new("RGB", (32, 24), color).save(t)
        row = dict(
            case_dir=str(path),
            task_id=task["task_id"],
            image_id=task["image_id"],
            prompt=task["instruction_en"],
            operation=task["operation"],
            status="ok",
            native_source_sha256=task["sha256"],
            seed=42,
            dtype="bfloat16",
            steps_requested=40,
            completed_steps=40,
            source_path=str(s),
            source_sha256=sha256(s),
            target_path=str(t),
            target={"sha256": sha256(t)},
            human_quality_label=None,
        )
        write(path / "result.json", row)
        if sealed:
            finish(path)
        return path

    return NS(cfg=cfg, tasks=tasks, code=code, out=out, legacy=legacy, case=case, root=tmp_path)


def test_real_manifest_is_one_unchanged_200_request_cohort():
    cfg = read(core.CONFIG)
    manifest = core.ROOT / cfg["manifest"]
    assert sha256(manifest) == cfg["manifest_sha256"]
    tasks = read(manifest)["tasks"]
    assert len(tasks) == 200 and len({t["task_id"] for t in tasks}) == 200
    assert len({t["image_id"] for t in tasks}) == 100
    assert tasks[:100] == read(core.ROOT / "configs/level3/qwen100-instructions-v1.json")["tasks"]
    assert tasks[100:] == read(core.ROOT / "configs/level3/qwen200-extension-instructions-v1.json")["tasks"]


def test_import_live_receipts_incremental_and_keep_originals(world):
    w = world
    old, new = w.legacy("old"), w.legacy("live")
    first = w.case(old, w.tasks[0])
    second = w.case(new, w.tasks[1], sealed=False)
    write(old / "results.json", [read(first / "result.json")])
    finish(old)
    hashes = {p: sha256(p) for p in old.rglob("*") if p.is_file()}
    (new / "results.json").write_text("{partial live write")
    (second / "receipt.json").write_text("{partial live write")
    state = core.collect(w.out, w.cfg, w.tasks, [old, new])
    assert state["successful"] == 1 and len(state["entries"]) == 4
    assert state["warnings"] and state["legacy_reviews"]
    assert read(w.out / "results.json")[1]["status"] == "not_run"
    # Simulate completion of the currently partial receipt, excluding that partial file.
    (second / "receipt.json").unlink()
    finish(second)
    state = core.collect(w.out, w.cfg, w.tasks, [old, new])
    assert state["successful"] == 2
    assert {p: sha256(p) for p in hashes} == hashes
    # Accepted evidence also survives omitting imports from a later resume command.
    assert core.collect(w.out, w.cfg, w.tasks, [])["successful"] == 2


@pytest.mark.parametrize("damage", ["image", "receipt", "prompt", "protocol", "conflict"])
def test_import_refuses_changed_or_conflicting_evidence(world, damage):
    w = world
    old = w.legacy("old")
    case = w.case(old, w.tasks[0])
    core.collect(w.out, w.cfg, w.tasks, [old])
    if damage == "image":
        (case / "target.png").write_bytes(b"tamper")
    elif damage == "receipt":
        core.atomic(case / "receipt.json", {"execution_complete": True})
    elif damage == "prompt":
        # A valid new receipt cannot override the task's immutable prompt.
        other = w.legacy("other")
        task = dict(w.tasks[1], instruction_en="a different edit")
        w.case(other, task)
        old = other
    elif damage == "protocol":
        meta = read(old / "run.json")
        meta["config"]["generation"]["seed"] = 123
        core.atomic(old / "run.json", meta)
    else:
        other = w.legacy("other")
        w.case(other, w.tasks[0], color="blue")
        old = other
    with pytest.raises(ValueError):
        core.collect(w.out, w.cfg, w.tasks, [old])


def test_workspace_resumes_same_folder_rejects_protocol_or_foreign_output(world):
    w = world
    assert core.workspace(w.out, w.cfg, w.code) == w.out
    changed = copy.deepcopy(w.cfg)
    changed["generation"]["seed"] = 99
    with pytest.raises(ValueError, match="protocol_changed"):
        core.workspace(w.out, changed, w.code)
    with pytest.raises(ValueError, match="outside_unified"):
        core.workspace(w.root / "runs/level3/old", w.cfg, w.code)
    foreign = core.RUNS / "foreign"
    foreign.mkdir()
    (foreign / "valuable.txt").write_text("keep")
    with pytest.raises(ValueError, match="nonempty"):
        core.workspace(foreign, w.cfg, w.code)


def test_shared_gpu_lock_fails_immediately_without_queue(tmp_path):
    with core.lock(tmp_path / "inference.lock"):
        with pytest.raises(RuntimeError, match="another_process_holds_lock"):
            with core.lock(tmp_path / "inference.lock"):
                pytest.fail("must never enter another inference")


def test_commit_gate_precedes_pipeline(world, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["unified", "infer", "--output", str(world.out)])
    monkeypatch.setattr(runner, "inputs", lambda: (world.cfg, world.tasks))
    monkeypatch.setattr(runner, "identity", lambda: (_ for _ in ()).throw(ValueError("uncommitted")))
    monkeypatch.setattr(runner, "infer_pending", lambda *a: pytest.fail("GPU reached before commit"))
    with pytest.raises(ValueError, match="uncommitted"):
        runner.main()


def fake_runtime():
    state = NS(calls=[], fail_at=None, steps=40, closes=0)

    class OutOfMemoryError(Exception):
        pass

    class Editor:
        def __init__(self, cfg, attempt):
            assert cfg["generation"]["seed"] == 42 and cfg["batch_size"] == 1

        def generate(self, source, prompt, target):
            state.calls.append(prompt)
            if len(state.calls) == state.fail_at:
                raise OutOfMemoryError("Synthetic CPU exception, not a real GPU OOM")
            Image.new("RGB", (32, 24), "purple").save(target)
            return dict(
                completed_steps=state.steps,
                target={"sha256": sha256(target)},
                inference_seconds=0,
                torch_peak_reserved_mib=0,
            )

        def close(self):
            state.closes += 1

    tracking = NS(id="CPU-only", url="synthetic://no-network", log=lambda _: None, finish=lambda **_: None)
    return state, Editor, lambda **_: tracking


def test_interrupted_resume_same_output_only_missing_and_preserves_attempts(world):
    w = world
    old = w.legacy("old")
    imported = w.case(old, w.tasks[0])
    original_receipt = sha256(imported / "receipt.json")
    state = core.collect(w.out, w.cfg, w.tasks, [old])
    fake, editor, tracker = fake_runtime()
    fake.fail_at = 2
    assert runner.infer_pending(w.out, w.cfg, w.tasks, state, w.code, editor, tracker) == 2
    assert state["successful"] == 2
    assert read(w.out / "attempts/attempt-0001/failure.json")["oom"] is True
    failed = w.out / "cases/task-2/attempt-0001"
    assert failed.exists() and not (failed / "receipt.json").exists()
    kept = {p: sha256(p) for p in failed.iterdir()}
    first_success = w.out / "cases/task-1/attempt-0001/receipt.json"
    first_success_hash = sha256(first_success)
    state = core.collect(w.out, w.cfg, w.tasks, [])
    fake.calls.clear()
    fake.fail_at = None
    assert runner.infer_pending(w.out, w.cfg, w.tasks, state, w.code, editor, tracker) == 0
    assert fake.calls == [t["instruction_en"] for t in w.tasks[2:]]
    assert state["execution_complete"] and state["successful"] == 4
    assert {p: sha256(p) for p in kept} == kept
    assert sha256(first_success) == first_success_hash
    assert sha256(imported / "receipt.json") == original_receipt
    assert (w.out / "cases/task-2/attempt-0002/receipt.json").exists()
    assert fake.closes == 2
    assert all(r["human_quality_label"] is None for r in read(w.out / "results.json"))
    assert os.path.samefile(imported / "source.png", w.out / "cases/task-2/attempt-0002/source.png")

    # Rerun when all complete must neither construct the editor nor contact W&B.
    def never(*a, **kw):
        pytest.fail("completed run must not start inference/tracking")

    assert runner.infer_pending(w.out, w.cfg, w.tasks, state, w.code, never, never) == 0


def test_39_steps_never_count_as_success(world):
    w = world
    state = core.collect(w.out, w.cfg, w.tasks, [])
    fake, editor, tracker = fake_runtime()
    fake.steps = 39
    assert runner.infer_pending(w.out, w.cfg, w.tasks, state, w.code, editor, tracker) == 2
    assert state["successful"] == 0
    assert not list((w.out / "cases").glob("*/attempt-*/receipt.json"))


def test_preview_cache_hardlinks_bindings_and_pending_labels(world):
    w = world
    old = w.legacy("old")
    case = w.case(old, w.tasks[0])
    state = core.collect(w.out, w.cfg, w.tasks, [old])
    viewer.render(w.out, w.cfg, w.tasks, state)
    data = read(w.out / "review-data.json")
    assert len(data["cases"]) == 4 and data["successful"] == 1
    a, pending = data["cases"][0], data["cases"][2]
    assert pending["binding"] is None and pending["target"] is None
    assert a["source"] == pending["source"]  # same original, two independent requests
    assert os.path.samefile(w.out / a["target"]["native"], case / "target.png")
    assert "edit_fulfillment" not in a  # no generated quality labels
    stats = {p: p.stat().st_mtime_ns for p in (w.out / "media").iterdir()}
    viewer.render(w.out, w.cfg, w.tasks, state)
    assert {p: p.stat().st_mtime_ns for p in stats} == stats
    assert read(w.out / "review-data.json")["cases"][0]["binding"] == a["binding"]


@pytest.fixture
def duplicate_runs(world, monkeypatch):
    w = world
    roots = [w.legacy("qwen100-v1"), w.legacy("qwen-react100-v1")]
    for r in roots:
        Image.new("RGB", (32, 24), "red").save(r / "image.png")
        write(r / "results.json", [])
        finish(r)
    monkeypatch.setattr(cleanup, "ROOTS", roots)
    monkeypatch.setattr(cleanup, "active_writers", lambda _: [])
    return w, roots


def test_cleanup_reclaims_only_exact_bytes_preserves_paths_receipts_and_is_idempotent(duplicate_runs):
    w, roots = duplicate_runs
    hashes = {p: sha256(p) for r in roots for p in r.iterdir()}
    pack = cleanup.plan(roots)
    assert len(pack["operations"]) == 1
    result = cleanup.apply(pack, w.out / "cleanup-journal.json")
    assert result["reclaimed_allocated_bytes"] > 0
    assert result["paths_deleted"] == 0
    assert {p: sha256(p) for p in hashes} == hashes
    assert os.path.samefile(roots[0] / "image.png", roots[1] / "image.png")
    assert not cleanup.plan(roots)["operations"]


@pytest.mark.parametrize("damage", ["keeper", "duplicate", "allowlist", "active"])
def test_cleanup_refuses_stale_or_active_plan(duplicate_runs, monkeypatch, damage):
    w, roots = duplicate_runs
    pack = cleanup.plan(roots)
    if damage in ("keeper", "duplicate"):
        from pathlib import Path

        Path(pack["operations"][0][damage]).write_bytes(b"changed since plan")
    elif damage == "allowlist":
        pack["roots"].append(str(w.root / "unrelated-user-data"))
    else:
        monkeypatch.setattr(cleanup, "active_writers", lambda _: [{"pid": 1}])
    with pytest.raises(ValueError):
        cleanup.apply(pack, w.out / "cleanup-journal.json")
