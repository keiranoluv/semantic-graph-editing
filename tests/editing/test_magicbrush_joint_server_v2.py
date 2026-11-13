"""Regression: a vLLM-only env must never import optional review/GQA modules."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools import magicbrush_joint_server_v2 as entrypoint
from tools.magicbrush_joint_v1.prepare import DEFAULT_CONFIG

ROOT = Path(__file__).resolve().parents[2]


def isolated_cli(stage):
    script = r"""
import importlib.abc
import json
import sys

class BlockOptionalReview(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = (
            "h5py", "pyarrow", "wandb", "torch_geometric", "src.datasets.gqa_full",
            "tools.review_magicbrush", "tools.magicbrush_joint_v1.review",
            "tools.magicbrush_joint_v1.runner", "tools.magicbrush_joint_v1.__main__",
        )
        if any(fullname == p or fullname.startswith(p + ".") for p in blocked):
            raise AssertionError("Server entrypoint imported optional module: " + fullname)

sys.meta_path.insert(0, BlockOptionalReview())
from tools.magicbrush_joint_v1 import server
server.runtime = lambda cfg: {"CPU_runtime_check": True}
def forbidden_serve(cfg):
    raise AssertionError("CPU command attempted to start GPU server")
server.serve = forbidden_serve
from tools.magicbrush_joint_server_v2 import main
main(["--stage", sys.argv[1]])
"""
    return subprocess.run(
        [sys.executable, "-c", script, stage],
        cwd=ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        check=True,
    )


def test_runtime_does_not_import_review_dependencies():
    result = isolated_cli("runtime")
    assert json.loads(result.stdout) == {"CPU_runtime_check": True}


def test_server_plan_does_not_import_review_dependencies_or_launch_gpu():
    result = json.loads(isolated_cli("server-plan").stdout)
    assert result["GPU_started"] is False
    assert "--model inferRouter/Qwen3.6-35B-A3B-NVFP4" in result["command"]
    assert "--max-num-seqs 1" in result["command"]


def test_help_needs_only_standard_library():
    result = subprocess.run(
        [sys.executable, "-S", "-m", "tools.magicbrush_joint_server_v2", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "serve-model" in result.stdout


def test_entrypoint_records_separate_provenance_without_changing_annotation_run(monkeypatch, tmp_path):
    from tools.magicbrush_joint_v1 import prepare

    cfg = json.loads(DEFAULT_CONFIG.read_text())
    run = {"code_revision": "a" * 40, "frozen": True}
    monkeypatch.setattr(prepare, "check_run", lambda config: (run, {}))
    monkeypatch.setattr(prepare, "revision", lambda: "b" * 40)
    result = entrypoint.launcher_provenance(cfg)
    assert result["schema_version"] == "magicbrush_joint_server_entrypoint_v2"
    assert result["annotation_code_revision"] == "a" * 40
    assert result["entrypoint_code_revision"] == "b" * 40
    assert len(result["entrypoint_sha256"]) == 64
    assert result["batch_size"] == 1 and result["seed"] == 42
    assert run == {"code_revision": "a" * 40, "frozen": True}


def test_annotation_stages_are_not_available_in_server_entrypoint():
    with pytest.raises(SystemExit) as error:
        entrypoint.main(["--stage", "infer"])
    assert error.value.code == 2
