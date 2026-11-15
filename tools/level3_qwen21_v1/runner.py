"""Bounded real Qwen 2.1 inference. No training, automatic labels or seed queue."""

import argparse
import fcntl
import importlib.metadata
import inspect
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

from tools.level3_image_graph_v1.core import ROOT, read, sha256, write

from .core import CACHE, CONFIG, LOCK, assets, bundle, code, finish, inputs, output, select
from .viewer import render


def runtime():
    import torch
    from diffusers import QwenImage21Pipeline

    dist = importlib.metadata.distribution("diffusers")
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    if direct.get("vcs_info", {}).get("commit_id") != read(CONFIG)["diffusers_commit"]:
        raise ValueError("Diffusers_must_be_installed_from_pinned_official_commit")
    required = {
        "image",
        "prompt",
        "num_inference_steps",
        "generator",
        "output_resolution",
        "use_kv_cache",
        "true_cfg_scale",
    }
    if not required <= set(inspect.signature(QwenImage21Pipeline.__call__).parameters):
        raise ValueError("official_image_editing_API_missing")
    if torch.version.cuda != "12.8" or "sm_120" not in torch.cuda.get_arch_list():
        # get_arch_list() is empty with CUDA_VISIBLE_DEVICES=''; inspect wheel flag in that case.
        if torch.version.cuda != "12.8" or "sm_120" not in torch._C._cuda_getArchFlags():
            raise ValueError("CUDA128_Blackwell_wheel_required")
    return QwenImage21Pipeline


class MemoryMonitor:
    """Sample total device usage, separately from PyTorch allocator counters."""

    def __init__(self):
        self.stop = threading.Event()
        self.samples = []
        self.thread = threading.Thread(target=self.poll, daemon=True)

    def poll(self):
        while not self.stop.is_set():
            try:
                line = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits", "-i", "0"],
                    text=True,
                    timeout=3,
                ).strip()
                used, total = map(int, line.split(","))
                self.samples.append({"monotonic": time.perf_counter(), "used_mib": used, "total_mib": total})
            except (subprocess.SubprocessError, ValueError, OSError):
                pass
            self.stop.wait(0.25)

    def start(self):
        self.thread.start()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=4)

    def summary(self, start=0):
        rows = [r for r in self.samples if r["monotonic"] >= start]
        return {
            "scope": "total_GPU_0_usage_including_other_processes",
            "sample_count": len(rows),
            "peak_used_mib": max((r["used_mib"] for r in rows), default=None),
            "total_mib": rows[-1]["total_mib"] if rows else None,
            "sampling_interval_seconds": 0.25,
        }


def call_args(cfg, source, generator):
    return {
        "image": source,
        "prompt": cfg["prompt"],
        "num_inference_steps": 40,
        "generator": generator,
        "num_images_per_prompt": 1,
        "output_resolution": cfg["output_resolution"],
        "true_cfg_scale": cfg["true_cfg_scale"],
        "use_kv_cache": cfg["use_kv_cache"],
    }


def check_image(path):
    with Image.open(path) as im:
        im.load()
        arr = np.asarray(im.convert("RGB"))
        if min(im.size) < 64 or arr.max() == arr.min():
            raise ValueError("degenerate_output_image")
        return {"width": im.width, "height": im.height, "mode": im.mode, "sha256": sha256(path)}


def paired_row(base, out):
    tid = base["task_id"]
    imdir = out / "images"
    imdir.mkdir(exist_ok=True)
    source = imdir / (tid + "-source.png")
    ip2p = imdir / (tid + "-ip2p.png")
    shutil.copyfile(base["source_path"], source)
    shutil.copyfile(base["target_path"], ip2p)
    return {
        "task_id": tid,
        "image_id": base["image_id"],
        "prompt": base["instruction_en"],
        "source_path": str(source),
        "source_sha256": sha256(source),
        "ip2p_path": str(ip2p),
        "ip2p_sha256": sha256(ip2p),
        "ip2p_editor": base["editor"],
        "qwen21_path": None,
        "status": "pending",
        "human_review_status": "pending",
    }


def run(args):
    cfg, index = inputs()
    assets()
    Pipeline = runtime()
    selected = select(cfg, index, args.mode, args.smoke_run)
    import torch

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16_CUDA_required")
    out = output(args.output)
    provenance = {
        "schema": "qwen21-run-v1",
        "mode": args.mode,
        "device": "cuda",
        "gpu": torch.cuda.get_device_name(0),
        "config": cfg,
        "config_sha256": sha256(CONFIG),
        "asset_lock_sha256": sha256(LOCK),
        "code": code(),
        "smoke_receipt_sha256": sha256(args.smoke_run / "receipt.json") if args.smoke_run else None,
        "input_policy": "exact_bytes_of_IP2P_source_png",
        "resolution_policy": "official_Qwen_output_resolution_1024; IP2P_max_side_512",
        "seed_policy": "literal_42_per_instruction; historical_IP2P_seed_is_different",
        "automatic_queue": False,
        "training": False,
        "vae_tiling": True,
    }
    write(out / "run.json", provenance)
    write(out / "environment.json", {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()})
    write(out / "gpu-before.json", {"nvidia_smi": subprocess.check_output(["nvidia-smi"], text=True)})
    rows = [paired_row(r, out) for r in selected]
    monitor = MemoryMonitor()
    monitor.start()
    started = time.perf_counter()
    load_seconds = None
    error = None
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.cuda.reset_peak_memory_stats()
    try:
        t0 = time.perf_counter()
        pipe = Pipeline.from_pretrained(
            CACHE / "model", torch_dtype=torch.bfloat16, local_files_only=True, use_safetensors=True
        )
        for component in (pipe.text_encoder, pipe.transformer, pipe.vae):
            component.eval().requires_grad_(False)
        pipe.enable_model_cpu_offload(gpu_id=0)
        pipe.vae.enable_tiling()
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - t0
        write(
            out / "pipeline.json",
            {
                "pipeline_class": type(pipe).__name__,
                "scheduler_class": type(pipe.scheduler).__name__,
                "scheduler": dict(pipe.scheduler.config),
                "pipeline_source_sha256": sha256(inspect.getfile(Pipeline)),
                "memory_mode": "model_cpu_offload",
                "dtype": "bfloat16",
            },
        )
        for row in rows:
            row["status"] = "running"
            case_start = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            steps = []

            def progress(_pipe, i, _t, kwargs, steps=steps, row=row):
                steps.append(i)
                if (i + 1) % 10 == 0:
                    print(json.dumps({"task_id": row["task_id"], "step": i + 1, "total": 40}), flush=True)
                return kwargs

            with Image.open(row["source_path"]) as im:
                source = im.convert("RGB")
                kwargs = call_args(
                    {**cfg, "prompt": row["prompt"]}, source, torch.Generator(device="cuda").manual_seed(42)
                )
                with torch.inference_mode():
                    result = pipe(**kwargs, callback_on_step_end=progress)
            torch.cuda.synchronize()
            dest = out / "images" / (row["task_id"] + "-qwen21.png")
            result.images[0].save(dest)
            row.update(
                qwen21_path=str(dest),
                inference_seconds=time.perf_counter() - case_start,
                torch_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                torch_peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                gpu_memory=monitor.summary(case_start),
                completed_steps=len(steps),
                qwen21_config={
                    k: cfg[k]
                    for k in [
                        "model",
                        "revision",
                        "dtype",
                        "seed",
                        "num_inference_steps",
                        "output_resolution",
                        "true_cfg_scale",
                        "use_kv_cache",
                        "memory_mode",
                    ]
                },
            )
            row["qwen21_image"] = check_image(dest)
            if len(steps) != 40:
                raise ValueError("incomplete_denoising_steps")
            row["status"] = "ok"
            write(out / (row["task_id"] + ".json"), row)
            print(
                json.dumps({"task_id": row["task_id"], "status": "ok", "seconds": row["inference_seconds"]}), flush=True
            )
    except (Exception, KeyboardInterrupt) as exc:
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "oom": isinstance(exc, torch.OutOfMemoryError),
            "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "torch_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        }
        for row in rows:
            if row["status"] == "running":
                row.update(status="oom" if error["oom"] else "error", error=repr(exc))
            elif row["status"] == "pending":
                row.update(status="not_run", error="preceding_failure")
        write(out / "failure.json", error)
    finally:
        monitor.close()
        write(out / "results.json", rows)
        write(
            out / "execution.json",
            {
                "model_load_seconds": load_seconds,
                "total_seconds": time.perf_counter() - started,
                "successful": sum(r["status"] == "ok" for r in rows),
                "attempted": len(rows),
                "error": error,
                "gpu_memory": monitor.summary(),
                "memory_samples": monitor.samples,
                "human_quality_measured": False,
            },
        )
        render(rows, out)
        finish(out, success=error is None)
    print(json.dumps({"output": str(out), "success": error is None}), flush=True)
    return 0 if error is None else 2


def compare(args):
    rows = []
    seen = set()
    for p in args.runs:
        for row in bundle(p, "results.json"):
            if row["task_id"] in seen:
                raise ValueError("duplicate_comparison_task")
            seen.add(row["task_id"])
            rows.append(row)
    if len(rows) > 6:
        raise ValueError("comparison_exceeds_bounded_pilot")
    out = output(args.output)
    write(out / "results.json", rows)
    write(
        out / "provenance.json",
        {"input_receipts": {str(p): sha256(p / "receipt.json") for p in args.runs}, "code": code()},
    )
    render(rows, out)
    finish(out)
    print(out / "comparison.html")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("preflight")
    sub.add_parser("prepare")
    for mode in ("smoke", "pilot"):
        p = sub.add_parser(mode)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--smoke-run", type=Path, required=mode == "pilot")
    p = sub.add_parser("compare")
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        from .assets import prepare

        prepare()
        return 0
    if args.mode == "preflight":
        cfg, index = inputs()
        assets()
        Pipeline = runtime()
        print(
            json.dumps(
                {
                    "status": "ok",
                    "pipeline": Pipeline.__name__,
                    "smoke_task": cfg["smoke_task"],
                    "max_extra_tasks": len(cfg["pilot_tasks"]),
                }
            )
        )
        return 0
    if args.mode == "compare":
        compare(args)
        return 0
    with (ROOT / "data/cache/level3-sgg-v1/inference.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run(args)
