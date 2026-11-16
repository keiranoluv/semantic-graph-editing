"""Frozen official Qwen backend shared by all 200 requests. Loaded only for missing targets."""

import importlib.metadata
import inspect
import time

from PIL import Image

from tools.level3_image_graph_v1.core import sha256, write
from tools.level3_qwen21_v1.core import CACHE, assets
from tools.level3_shared.qwen_runtime import MemoryMonitor, call_args, check_image, runtime


class Editor:
    def __init__(self, cfg, attempt):
        assets()
        Pipeline = runtime()
        import torch

        self.torch, self.cfg = torch, cfg
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16_CUDA_required")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        self.monitor = MemoryMonitor()
        self.monitor.start()
        self.pipe = None
        try:
            t0 = time.perf_counter()
            self.pipe = Pipeline.from_pretrained(
                CACHE / "model", torch_dtype=torch.bfloat16, local_files_only=True, use_safetensors=True
            )
            for part in [self.pipe.text_encoder, self.pipe.transformer, self.pipe.vae]:
                part.eval().requires_grad_(False)
            self.pipe.enable_model_cpu_offload(gpu_id=0)
            self.pipe.vae.enable_tiling()
            self.pipe.set_progress_bar_config(disable=True)
            write(
                attempt / "pipeline.json",
                {
                    "class": type(self.pipe).__name__,
                    "load_seconds": time.perf_counter() - t0,
                    "scheduler": dict(self.pipe.scheduler.config),
                    "pipeline_source_sha256": sha256(inspect.getfile(Pipeline)),
                    "device": torch.cuda.get_device_name(0),
                    "memory_mode": "model_cpu_offload",
                    "vae_tiling": True,
                },
            )
            write(
                attempt / "environment.json",
                {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
            )
        except BaseException:
            self.close()
            raise

    def generate(self, source, prompt, target):
        torch = self.torch
        steps = []

        def progress(_p, i, _t, kwargs):
            steps.append(i)
            return kwargs

        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        with Image.open(source) as im, torch.inference_mode():
            result = self.pipe(
                **call_args(
                    {**self.cfg["generation"], "prompt": prompt},
                    im.convert("RGB"),
                    torch.Generator("cuda").manual_seed(42),
                ),
                callback_on_step_end=progress,
            )
        torch.cuda.synchronize()
        result.images[0].save(target)
        if len(steps) != 40:
            raise ValueError("incomplete_diffusion_steps")
        return {
            "target": check_image(target),
            "completed_steps": len(steps),
            "inference_seconds": time.perf_counter() - started,
            "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "torch_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            "gpu_memory": self.monitor.summary(started),
        }

    def close(self):
        self.monitor.close()
        self.pipe = None
