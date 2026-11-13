"""CPU regressions for venv PATH, CUDA toolkit layout and sampling preflight."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools import magicbrush_joint_server_v3 as server
from tools.magicbrush_joint_v1.storage import read


@pytest.fixture
def configured(tmp_path, monkeypatch):
    cuda = tmp_path / "cuda"
    for name in ("bin", "include", "lib", "nvvm"):
        (cuda / name).mkdir(parents=True)
    for name in ("bin/nvcc", "include/cuda_runtime.h", "lib/libcudart.so.13"):
        (cuda / name).write_bytes(b"CPU fixture")
    venv = tmp_path / "venv/bin"
    venv.mkdir(parents=True)
    (venv / "ninja").write_bytes(b"CPU fixture ninja")
    driver = tmp_path / "driver"
    driver.mkdir()
    (driver / "libcuda.so.1").write_bytes(b"CPU fixture driver")
    monkeypatch.setattr(server.os, "environ", dict(os.environ))
    monkeypatch.setattr(server.sys, "executable", str(venv / "python"))
    monkeypatch.setattr(server, "pip_cuda_root", lambda: cuda)
    monkeypatch.setattr(server, "cuda_library_dirs", lambda: [str(driver)])
    monkeypatch.setattr(server.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(server.importlib.metadata, "version", lambda name: "13.0.88")
    return tmp_path / "output", cuda, venv, driver


def test_environment_exposes_venv_ninja_and_cuda_library_aliases(configured):
    output, cuda, venv, driver = configured
    record = server.environment(output)
    assert os.environ["PATH"].split(os.pathsep)[:2] == [str(venv), str(cuda / "bin")]
    assert record["ninja"] == str(venv / "ninja")
    home = Path(os.environ["CUDA_HOME"])
    assert (home / "bin").resolve() == cuda / "bin"
    assert (home / "include").resolve() == cuda / "include"
    assert (home / "lib64/libcudart.so").resolve() == cuda / "lib/libcudart.so.13"
    assert (home / "lib64/libcuda.so").resolve() == driver / "libcuda.so.1"
    assert server.environment(output) == record
    assert os.environ["MAX_JOBS"] == "2"


def test_missing_ninja_is_detected_before_server_start(configured):
    output, _, venv, _ = configured
    (venv / "ninja").unlink()
    with pytest.raises(RuntimeError, match="venv_ninja_missing"):
        server.environment(output)


def test_cuda_toolchain_minor_mismatch_is_detected(configured, monkeypatch):
    output, _, _, _ = configured
    monkeypatch.setattr(server.importlib.metadata, "version", lambda name: "13.5.37")
    with pytest.raises(ValueError, match="mismatched_CUDA130_toolchain"):
        server.environment(output)


def test_symlink_refuses_to_overwrite_existing_evidence(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    first.touch()
    second.touch()
    link = tmp_path / "alias"
    server.symlink_once(link, first)
    with pytest.raises(ValueError, match="toolchain_link_changed"):
        server.symlink_once(link, second)
    with pytest.raises(ValueError, match="already_exists"):
        server.symlink_once(first, second)


def test_sampling_preflight_builds_without_loading_library(configured, monkeypatch, tmp_path):
    from tools import magicbrush_joint_toolchain_v1

    output, _, _, _ = configured
    record = server.environment(output)
    workspace = tmp_path / "workspace"
    monkeypatch.setenv("FLASHINFER_WORKSPACE_BASE", str(workspace))
    original = workspace / ".cache/flashinfer/13.0.88/120f/cached_ops/sampling/build.ninja"
    original.parent.mkdir(parents=True)
    original.write_text(f"cuda_home = /usr/local/cuda\nbuild {original.parent}/sampling.so: link\n")
    monkeypatch.setattr(magicbrush_joint_toolchain_v1, "check", lambda: {"CPU_host_compile": "PASS"})
    calls = []

    def cpu_build(command, **kwargs):
        calls.append(command)
        assert kwargs == {"check": True}
        assert command[0] == record["ninja"] and command[-2:] == ["-j", "2"]
        folder = Path(command[command.index("-C") + 1])
        (folder / "sampling.so").write_bytes(b"synthetic un-loadable CPU fixture")

    monkeypatch.setattr(server.subprocess, "run", cpu_build)
    result = server.preflight(output, record)
    assert len(calls) == 1
    assert result["CPU_compile_and_link"] == "PASS"
    assert result["GPU_started"] is False and result["shared_library_loaded"] is False
    proof = read(output / "preflights/flashinfer-cpu-build-v1.json")
    assert proof == result
    recipe = Path(calls[0][calls[0].index("-f") + 1]).read_text()
    assert record["environment"]["CUDA_HOME"] in recipe
    assert str(original.parent) not in recipe


def test_missing_prior_sampling_recipe_is_reported(configured, monkeypatch, tmp_path):
    from tools import magicbrush_joint_toolchain_v1

    output, _, _, _ = configured
    record = server.environment(output)
    monkeypatch.setenv("FLASHINFER_WORKSPACE_BASE", str(tmp_path / "empty"))
    monkeypatch.setattr(magicbrush_joint_toolchain_v1, "check", lambda: {})
    with pytest.raises(RuntimeError, match="missing_failed_run_sampling_build"):
        server.preflight(output, record)


def test_v3_help_requires_only_standard_library():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-S", "-m", "tools.magicbrush_joint_server_v3", "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "preflight" in result.stdout and "serve-model" in result.stdout
