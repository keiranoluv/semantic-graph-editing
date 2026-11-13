"""Versioned server environment repair: venv PATH and pip CUDA toolkit layout.

preflight builds FlashInfer's already-generated sampling extension on CPU only.
serve-model is user-invoked and keeps the frozen v1 model/annotation protocol.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from tools.magicbrush_joint_toolchain_v1 import cuda_library_dirs


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def symlink_once(path, target):
    path, target = Path(path), Path(target).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        if path.resolve() != target:
            raise ValueError(f"toolchain_link_changed:{path}")
    elif path.exists():
        raise ValueError(f"toolchain_path_already_exists:{path}")
    else:
        path.symlink_to(target, target_is_directory=target.is_dir())


def pip_cuda_root():
    distribution = importlib.metadata.distribution("nvidia-cuda-nvcc")
    for file in distribution.files or []:
        if str(file).endswith("/bin/nvcc"):
            return Path(distribution.locate_file(file)).resolve().parent.parent
    raise RuntimeError("pip_cuda_nvcc_missing")


def environment(output):
    """Create a local compatibility view; never modify system CUDA or pip files."""
    from tools.magicbrush_joint_v1.storage import write_once

    output = Path(output)
    root = pip_cuda_root()
    view = output / "toolchains/pip-cuda130-view-v1"
    bindir = Path(sys.executable).parent
    ninja = bindir / "ninja"
    if not ninja.is_file():
        raise RuntimeError(f"venv_ninja_missing:{ninja}")
    if not (root / "include/cuda_runtime.h").is_file():
        raise RuntimeError("pip_cuda_runtime_headers_missing")
    for name in ("bin", "include", "nvvm"):
        if (root / name).is_dir():
            symlink_once(view / name, root / name)
    libraries = root / "lib"
    for path in sorted(libraries.iterdir()):
        symlink_once(view / "lib64" / path.name, path)
    runtime = libraries / "libcudart.so.13"
    if not runtime.is_file():
        raise RuntimeError("pip_cuda13_runtime_library_missing")
    symlink_once(view / "lib64/libcudart.so", runtime)
    driver = Path(cuda_library_dirs()[0]) / "libcuda.so.1"
    symlink_once(view / "lib64/libcuda.so", driver)
    compiler, cxx = shutil.which("gcc"), shutil.which("g++")
    if not compiler or not cxx:
        raise RuntimeError("gcc_or_g++_missing")
    updates = {"CUDA_HOME": str(view), "CC": compiler, "CXX": cxx, "MAX_JOBS": "2"}
    # Absolute Python does not activate its venv's PATH. Put both tools first.
    prefixes = [str(bindir), str(root / "bin")]
    os.environ["PATH"] = os.pathsep.join([*prefixes, os.environ.get("PATH", "")])
    os.environ.update(updates)
    versions = {}
    for package in (
        "ninja",
        "nvidia-cuda-nvcc",
        "nvidia-cuda-runtime",
        "nvidia-cuda-crt",
        "nvidia-nvvm",
        "flashinfer-python",
    ):
        versions[package] = importlib.metadata.version(package)
    for package in ("nvidia-cuda-nvcc", "nvidia-cuda-crt", "nvidia-nvvm"):
        if versions[package] != "13.0.88":
            raise ValueError(f"mismatched_CUDA130_toolchain:{package}={versions[package]}")
    record = {
        "schema_version": "magicbrush_joint_server_environment_v3",
        "source_sha256": checksum(__file__),
        "pip_cuda_root": str(root),
        "path_prefix": prefixes,
        "environment": updates,
        "ninja": str(ninja),
        "nvcc": str(root / "bin/nvcc"),
        "cuda_runtime_library": str(runtime),
        "cuda_driver_library": str(driver.resolve()),
        "package_versions": versions,
    }
    write_once(view / "manifest.json", record)
    return record


def preflight(output, record):
    """Run Ninja/NVCC build only; never load the resulting sampling.so."""
    from tools.magicbrush_joint_toolchain_v1 import check
    from tools.magicbrush_joint_v1.storage import read, write_once

    output = Path(output)
    host = check()
    version = record["package_versions"]["flashinfer-python"]
    workspace = Path(os.environ.get("FLASHINFER_WORKSPACE_BASE", Path.home()))
    original = workspace / ".cache/flashinfer" / version / "120f/cached_ops/sampling/build.ninja"
    if not original.is_file():
        raise RuntimeError(
            f"missing_failed_run_sampling_build:{original}; preflight is scoped to this recorded RTX5090 failure"
        )
    directory = output / "toolchains/flashinfer-sampling-cuda130-build-v1"
    directory.mkdir(parents=True, exist_ok=True)
    build = re.sub(
        r"^cuda_home = .*", "cuda_home = " + record["environment"]["CUDA_HOME"], original.read_text(), flags=re.M
    )
    build = build.replace(str(original.parent), str(directory))
    ninja_file = directory / "build.ninja"
    if ninja_file.exists() and ninja_file.read_text() != build:
        raise ValueError("sampling_build_recipe_changed; use a new toolchain version")
    if not ninja_file.exists():
        ninja_file.write_text(build)
    subprocess.run([record["ninja"], "-v", "-C", str(directory), "-f", str(ninja_file), "-j", "2"], check=True)
    shared = directory / "sampling.so"
    if not shared.is_file():
        raise RuntimeError("sampling_build_output_missing")
    result = {
        "schema_version": "magicbrush_joint_flashinfer_cpu_build_v1",
        "server_environment": record,
        "host_toolchain": host,
        "ninja_recipe_sha256": checksum(ninja_file),
        "sampling_shared_library": str(shared),
        "sampling_shared_library_sha256": checksum(shared),
        "sources_sha256": {str(path): checksum(path) for path in [original.parent / "build.ninja"]},
        "CPU_compile_and_link": "PASS",
        "shared_library_loaded": False,
        "GPU_started": False,
    }
    receipt = output / "preflights/flashinfer-cpu-build-v1.json"
    if receipt.exists():
        previous = read(receipt)
        # The upstream cache recipe may be regenerated by a later server start.
        # Keep its initial hash when our effective CPU build recipe is unchanged.
        result["sources_sha256"] = previous["sources_sha256"]
    write_once(receipt, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--stage", required=True, choices=("runtime", "preflight", "server-plan", "serve-model"))
    args = parser.parse_args(argv)
    from tools.magicbrush_joint_v1.prepare import DEFAULT_CONFIG, check_run, config, revision
    from tools.magicbrush_joint_v1.server import runtime, serve, server_args
    from tools.magicbrush_joint_v1.storage import exclusive, read, write_once
    from tools.prepare_magicbrush import ROOT, digest

    cfg = config(args.config or DEFAULT_CONFIG)
    run, _ = check_run(cfg)
    output = ROOT / cfg["output"]
    with exclusive(output):
        record = environment(output)
        if args.stage == "preflight":
            result = preflight(output, record)
        elif args.stage == "runtime":
            result = {**runtime(cfg), "server_environment": record, "GPU_started": False}
        elif args.stage == "server-plan":
            result = {"server_environment": record, "command": shlex.join(server_args(cfg)), "GPU_started": False}
        else:
            proof = read(output / "preflights/flashinfer-cpu-build-v1.json")
            if (
                proof["server_environment"] != record
                or checksum(proof["sampling_shared_library"]) != proof["sampling_shared_library_sha256"]
            ):
                raise ValueError("CPU_sampling_build_provenance_changed")
            provenance = {
                "schema_version": "magicbrush_joint_server_entrypoint_v3",
                "source_sha256": checksum(__file__),
                "code_revision": revision(),
                "annotation_run_sha256": digest(run),
                "server_environment": record,
                "CPU_preflight_sha256": digest(proof),
                "automatic_queue_continuation": False,
            }
            path = output / "launchers/server-entrypoint-v3.json"
            if path.exists():
                provenance["code_revision"] = read(path)["code_revision"]
            write_once(path, provenance)
    if args.stage == "serve-model":
        raise SystemExit(serve(cfg))  # Only the user invokes the GPU server.
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
