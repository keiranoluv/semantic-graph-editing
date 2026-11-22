"""Unchanged Qwen runtime contracts shared with the REACT++ memory monitor."""

import importlib.metadata
import inspect
import json
import subprocess
import threading
import time

import numpy as np
from PIL import Image
from task3_vn100.image_graph import read, sha256
from task3_vn100.qwen_assets import CONFIG


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
