"""Explicit user-invoked server launcher and CPU runtime compatibility inspection."""

import importlib.metadata
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from src.knowledge.transe.core import sha256
from tools.magicbrush_joint_v1.prepare import check_run
from tools.magicbrush_joint_v1.storage import exclusive, write_once
from tools.prepare_magicbrush import ROOT, digest


def server_args(cfg):
    model, server = cfg["model"], cfg["server"]
    return [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model["id"],
        "--revision",
        model["revision"],
        "--served-model-name",
        server["served_model_name"],
        "--host",
        server["host"],
        "--port",
        str(server["port"]),
        "--tensor-parallel-size",
        "1",
        "--max-num-seqs",
        "1",
        "--max-model-len",
        str(server["max_model_len"]),
        "--max-num-batched-tokens",
        str(server["max_num_batched_tokens"]),
        "--gpu-memory-utilization",
        str(server["gpu_memory_utilization"]),
        "--dtype",
        "bfloat16",
        "--quantization",
        model["quantization"],
        "--limit-mm-per-prompt",
        '{"image":2,"video":0}',
        "--mm-processor-kwargs",
        json.dumps(server["mm_processor_kwargs"]),
        "--seed",
        str(cfg["seed"]),
        "--generation-config",
        "vllm",
        *server["extra_args"],
    ]


def runtime(cfg):
    versions = {}
    for package in ("vllm", "torch", "transformers", "compressed-tensors", "flashinfer-python", "safetensors"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    if versions["vllm"] != cfg["server"]["vllm_version"]:
        raise ValueError(f"vllm_version_mismatch: {versions['vllm']}; expected {cfg['server']['vllm_version']}")
    spec = importlib.util.find_spec("vllm")
    root = Path(spec.origin).parent
    files = [
        "model_executor/models/registry.py",
        "model_executor/layers/quantization/compressed_tensors/compressed_tensors.py",
        "model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a4_nvfp4.py",
    ]
    if cfg["model"]["architecture"] not in (root / files[0]).read_text():
        raise ValueError("vllm_architecture_not_registered")
    # Hash the installed model implementation too; alternate compressed-tensors
    # VLMs can be selected explicitly without changing observation/compiler code.
    model_files = {
        "Qwen3_5MoeForConditionalGeneration": "qwen3_5.py",
        "Qwen3VLForConditionalGeneration": "qwen3_vl.py",
        "InternVLChatModel": "internvl.py",
    }
    architecture = cfg["model"]["architecture"]
    if architecture not in model_files:
        raise ValueError("runtime_model_source_not_verified; add a versioned compatibility check")
    files.append("model_executor/models/" + model_files[architecture])
    if cfg["model"]["quantization"] != "compressed-tensors":
        raise ValueError("This runtime checker currently verifies compressed-tensors; version it for another backend")
    return {"versions": versions, "runtime_source_sha256": {p: sha256(root / p) for p in files}}


def serve(cfg):
    run, _ = check_run(cfg)
    info = runtime(cfg)  # Metadata/source checks only; GPU initialization happens in the subprocess.
    from tools.magicbrush_joint_v1.compatibility import verify_model

    model_evidence = verify_model(cfg)
    receipt = {
        "run_sha256": digest(run),
        "config_sha256": digest(cfg),
        **info,
        "model_evidence": model_evidence,
        "argv": server_args(cfg),
    }
    output = ROOT / cfg["output"]
    # Same advisory lock used by existing graph-model launchers, held for server lifetime.
    with exclusive(ROOT / "runs/level2/.gqa-kg-training-device"):
        write_once(output / "server.json", receipt)
        return subprocess.call(receipt["argv"], cwd=ROOT)
