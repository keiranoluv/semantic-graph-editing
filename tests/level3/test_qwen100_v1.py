import copy
import hashlib

import pytest
from PIL import Image

from tools.level3_image_graph_v1.core import sha256, write
from tools.level3_qwen21_v1.core import bundle, finish
from tools.level3_qwen100_v1 import core
from tools.level3_qwen100_v1.receipt import finish_run
from tools.level3_qwen100_v1.viewer import render


def fixture_pack(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "ROOT", tmp_path)
    return {
        "source_revisions": {"sidewalk": "a" * 40, "travel": "b" * 40},
        "tasks": [
            {
                "task_id": f"t{i}",
                "image_id": f"i{i}",
                "dataset": "sidewalk" if i < 50 else "travel",
                "sha256": hashlib.sha256(str(i).encode()).hexdigest(),
                "dhash": hashlib.sha256(str(i).encode()).hexdigest()[:16],
                "source_path": str(tmp_path / "data/raw/level3/qwen21-batch100-v1" / f"{i}.jpg"),
                "source_revision_sha": ("a" if i < 50 else "b") * 40,
                "author": "assistant_visual_review",
                "human_outcome_label": None,
                "instruction_en": "Add a cat beside the tree.",
                "visual_evidence": "Tree beside empty pavement",
                "operation": {"kind": "add", "label": "cat", "location": "beside tree"},
            }
            for i in range(100)
        ],
    }


def test_exact_unique_cohort(tmp_path, monkeypatch):
    p = fixture_pack(tmp_path, monkeypatch)
    assert len(core.validate(p, False)) == 100
    p["tasks"].pop()
    with pytest.raises(ValueError, match="exact_100"):
        core.validate(p, False)


def test_duplicate_source_rejected(tmp_path, monkeypatch):
    p = fixture_pack(tmp_path, monkeypatch)
    p["tasks"][1]["sha256"] = p["tasks"][0]["sha256"]
    with pytest.raises(ValueError, match="duplicate_sha256"):
        core.validate(p, False)


def test_near_duplicate_rejected(tmp_path, monkeypatch):
    p = fixture_pack(tmp_path, monkeypatch)
    p["tasks"][1]["dhash"] = p["tasks"][0]["dhash"]
    with pytest.raises(ValueError, match="near_duplicate"):
        core.validate(p, False)


def test_no_forged_human_label(tmp_path, monkeypatch):
    p = fixture_pack(tmp_path, monkeypatch)
    p["tasks"][0]["human_outcome_label"] = True
    with pytest.raises(ValueError, match="human_annotation"):
        core.validate(p, False)


def test_wrong_source_revision(tmp_path, monkeypatch):
    p = fixture_pack(tmp_path, monkeypatch)
    p["tasks"][0]["source_revision_sha"] = "b" * 40
    with pytest.raises(ValueError, match="unpinned_source"):
        core.validate(p, False)


def test_no_overwrite_or_locked_output(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "RUNS", tmp_path / "runs")
    path = core.new_output(tmp_path / "runs/new")
    with pytest.raises(FileExistsError):
        core.new_output(path)
    with pytest.raises(ValueError):
        core.new_output(tmp_path / "locked")


def test_normalize_input_preserves_original(tmp_path):
    p = tmp_path / "source.jpg"
    Image.new("RGB", (1600, 900), "green").save(p)
    before = sha256(p)
    dest = tmp_path / "new.png"
    core.source_png({"source_path": str(p)}, dest)
    with Image.open(dest) as im:
        assert im.size == (1600, 900)
    assert sha256(p) == before


def test_resume_only_bound_successes_and_never_edits_old(tmp_path):
    ident = {
        "manifest_sha256": "m",
        "config_sha256": "c",
        "asset_lock_sha256": "a",
        "code": {"files_sha256": {"runner": "x"}},
    }
    write(tmp_path / "run.json", ident)
    case = tmp_path / "cases/t1"
    case.mkdir(parents=True)
    write(case / "result.json", {"task_id": "t1", "status": "ok", "case_dir": str(case)})
    finish(case)
    before = sha256(case / "receipt.json")
    assert list(core.reuse(tmp_path, ident)) == ["t1"]
    assert sha256(case / "receipt.json") == before
    changed = copy.deepcopy(ident)
    changed["manifest_sha256"] = "wrong"
    with pytest.raises(ValueError, match="resume_protocol"):
        core.reuse(tmp_path, changed)
    (case / "result.json").write_text("{}")
    with pytest.raises(ValueError, match="changed_artifact"):
        core.reuse(tmp_path, ident)


def test_receipt_excludes_mutable_tracking(tmp_path):
    write(tmp_path / "results.json", [])
    (tmp_path / "wandb").mkdir()
    (tmp_path / "wandb/transport").write_text("x")
    finish_run(tmp_path, True)
    (tmp_path / "wandb/transport").write_text("changed")
    assert bundle(tmp_path, "results.json") == []


def test_viewer_never_autolabels(tmp_path):
    im = tmp_path / "source.png"
    Image.new("RGB", (64, 64), "green").save(im)
    render([{"task_id": "t", "source_path": str(im), "prompt": "<script>bad</script>", "status": "ok"}], tmp_path)
    text = (tmp_path / "comparison.html").read_text()
    assert "<script>bad</script>" not in text
    import json

    template = json.loads((tmp_path / "review-template.json").read_text())
    assert template["reviews"][0]["edit_fulfillment"] is None
