"""CPU toolchain regressions; never load a shared object or initialize CUDA."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import magicbrush_joint_toolchain_v1 as preflight


def test_missing_compiler_fails_before_importing_triton(monkeypatch):
    monkeypatch.delenv("CC", raising=False)
    monkeypatch.setattr(preflight.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="missing_C_compiler"):
        preflight.check()


def test_explicit_cc_is_honored(monkeypatch):
    monkeypatch.setenv("CC", "/custom/gcc")
    monkeypatch.setattr(preflight.shutil, "which", lambda name: name if name == "/custom/gcc" else None)
    assert preflight.compiler_path() == "/custom/gcc"


def test_missing_python_headers_fail_early(monkeypatch, tmp_path):
    monkeypatch.setattr(preflight.sysconfig, "get_paths", lambda **kwargs: {"include": str(tmp_path)})
    with pytest.raises(RuntimeError, match="missing_Python_headers"):
        preflight.python_include()


def test_explicit_cuda_library_path(monkeypatch, tmp_path):
    monkeypatch.setenv("TRITON_LIBCUDA_PATH", str(tmp_path))
    with pytest.raises(RuntimeError, match="missing_libcuda"):
        preflight.cuda_library_dirs()
    (tmp_path / "libcuda.so.1").write_bytes(b"test-fixture")
    assert preflight.cuda_library_dirs() == [str(tmp_path.resolve())]


def test_compile_does_not_load_shared_library_or_launch_gpu(monkeypatch, tmp_path):
    nvidia = tmp_path / "backends/nvidia"
    nvidia.mkdir(parents=True)
    (nvidia / "driver.c").write_text("/* synthetic test source */")
    monkeypatch.setattr(preflight, "compiler_path", lambda: "/usr/bin/gcc")
    monkeypatch.setattr(preflight, "python_include", lambda: tmp_path / "python-include")
    monkeypatch.setattr(preflight, "cuda_library_dirs", lambda: [str(tmp_path / "cuda-lib")])
    monkeypatch.setattr(
        preflight.importlib.util, "find_spec", lambda name: SimpleNamespace(origin=str(tmp_path / "__init__.py"))
    )
    monkeypatch.setattr(preflight.importlib.metadata, "version", lambda name: "test-triton")
    calls = []

    def compile_only(command, **kwargs):
        calls.append(command)
        assert kwargs == {"check": True, "capture_output": True, "text": True}
        assert command[0] == "/usr/bin/gcc" and "-shared" in command
        assert "-l:libcuda.so.1" in command
        assert str(nvidia / "driver.c") in command
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(preflight.subprocess, "run", compile_only)
    monkeypatch.setattr(preflight.subprocess, "check_output", lambda *args, **kwargs: "gcc fixture\n")
    result = preflight.check()
    assert len(calls) == 1
    assert result["triton_host_helper_compile_and_link"] == "PASS"
    assert result["GPU_started"] is False and result["compiled_library_loaded"] is False
    output = Path(calls[0][calls[0].index("-o") + 1])
    assert not output.parent.exists()  # Temporary compilation outputs are removed.
