"""CPU-only compiler/header/linker preflight for the local joint-VLM server.

Compile Triton's actual CUDA host helper into a temporary shared library.
Do not load that library, initialize CUDA, load weights, or execute GPU kernels.
The frozen annotation/server sources and receipts remain unchanged.
"""

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path


def compiler_path():
    requested = os.environ.get("CC")
    compiler = shutil.which(requested) if requested else (shutil.which("gcc") or shutil.which("clang"))
    if compiler is None:
        raise RuntimeError("missing_C_compiler: install gcc/clang or set CC to its executable path")
    return compiler


def python_include():
    scheme = sysconfig.get_default_scheme()
    if scheme == "posix_local":
        scheme = "posix_prefix"
    path = Path(sysconfig.get_paths(scheme=scheme)["include"])
    if not (path / "Python.h").is_file():
        raise RuntimeError(f"missing_Python_headers:{path}/Python.h; install the matching python3.X-dev package")
    return path


def cuda_library_dirs():
    explicit = os.environ.get("TRITON_LIBCUDA_PATH")
    if explicit:
        candidates = [explicit]
    else:
        output = subprocess.check_output(["/sbin/ldconfig", "-p"], text=True)
        candidates = [str(Path(line.split()[-1]).parent) for line in output.splitlines() if "libcuda.so.1" in line]
        if not candidates:
            candidates = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    found = sorted({str(Path(p).resolve()) for p in candidates if p and (Path(p) / "libcuda.so.1").is_file()})
    if not found:
        raise RuntimeError("missing_libcuda.so.1: CUDA driver library must be visible to the server container")
    return found


def check():
    compiler = compiler_path()
    headers = python_include()
    libraries = cuda_library_dirs()
    spec = importlib.util.find_spec("triton")
    if spec is None or spec.origin is None:
        raise RuntimeError("missing_triton: run this preflight with the vLLM environment Python")
    nvidia = Path(spec.origin).parent / "backends/nvidia"
    source = nvidia / "driver.c"
    if not source.is_file():
        raise RuntimeError(f"missing_triton_host_helper:{source}")
    with tempfile.TemporaryDirectory(prefix="magicbrush-toolchain-") as tmp:
        # Match triton.runtime.build._build, including Python and CUDA headers.
        command = [
            compiler,
            str(source),
            "-O3",
            "-shared",
            "-fPIC",
            "-Wno-psabi",
            "-o",
            str(Path(tmp) / "cuda_utils.so"),
            "-l:libcuda.so.1",
            *(f"-L{p}" for p in libraries),
            f"-I{nvidia / 'include'}",
            f"-I{headers}",
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        # The compiled .so is never imported or called. It is deleted here.
    version = subprocess.check_output([compiler, "--version"], text=True).splitlines()[0]
    return {
        "schema_version": "magicbrush_joint_toolchain_preflight_v1",
        "preflight_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "compiler": compiler,
        "compiler_version": version,
        "python": sys.version.split()[0],
        "python_include": str(headers),
        "triton_version": importlib.metadata.version("triton"),
        "triton_driver_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "cuda_library_dirs": libraries,
        "triton_host_helper_compile_and_link": "PASS",
        "compiled_library_loaded": False,
        "GPU_started": False,
        "interpretation": "CPU host toolchain check only; GPU runtime and image inference remain unverified.",
    }


def main():
    try:
        print(json.dumps(check(), indent=2))
    except subprocess.CalledProcessError as error:
        print(error.stderr or str(error), file=sys.stderr)
        raise SystemExit(1) from error
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
