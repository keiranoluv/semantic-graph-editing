"""Use the same pinned official REACT++ ONNX backend and CUDA profile as the historical run."""

import importlib.metadata
import random
import time

import numpy as np

from tools.level3_image_graph_v1.core import write
from tools.level3_qwen21_v1.runner import MemoryMonitor
from tools.level3_qwen_graphs_v1.core import verify_react
from tools.level3_sgg_pretrained_v2.assets import CACHE
from tools.level3_sgg_pretrained_v2.backends import React


def preflight():
    assets = verify_react()
    import onnxruntime as ort
    import torch

    if torch.version.cuda != "12.8" or "sm_120" not in torch._C._cuda_getArchFlags():
        raise ValueError("CUDA128_Blackwell_wheel_required")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise ValueError("ORT_CUDA_provider_unavailable")
    return assets


class Editor:
    def __init__(self, cfg, attempt):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA_required_no_CPU_model_fallback")
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.monitor = MemoryMonitor()
        self.monitor.start()
        try:
            self.model = React(CACHE, "cuda", attempt, cfg["react"])
            write(attempt / "model-metadata.json", self.model.metadata)
            write(
                attempt / "environment.json",
                {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
            )
            write(
                attempt / "runtime.json",
                {
                    "providers": self.model.sess.get_providers(),
                    "device": torch.cuda.get_device_name(),
                    "cuda_build": torch.version.cuda,
                    "seed": 42,
                    "batch_size": 1,
                    "preprocess": cfg["preprocess"],
                    "normalization": cfg["normalization"],
                },
            )
        except BaseException:
            self.monitor.close()
            raise

    def predict(self, path, native_path):
        started = time.perf_counter()
        objects, relations, native = self.model.predict(path, native_path)
        return (
            objects,
            relations,
            native,
            {"elapsed_seconds": time.perf_counter() - started, "gpu_memory": self.monitor.summary(started)},
        )

    def close(self):
        try:
            profile = self.model.close()
        finally:
            self.monitor.close()
        return {"profile": profile, "gpu_memory": self.monitor.summary()}
