"""Official ONNX models, original preprocessing, explicit CUDA execution evidence."""

import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np
from task3_vn100.models.graph_adapter import react_decode


def session(path, device, profile):
    import onnxruntime as ort

    if device == "cuda":
        import torch  # Preload wheel CUDA 12.8 / cuDNN libraries.

        if not torch.cuda.is_available() or "sm_120" not in torch.cuda.get_arch_list():
            raise RuntimeError("CUDA_or_Blackwell_kernels_unavailable")
        ort.preload_dlls()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.enable_profiling = True
    opts.profile_file_prefix = str(profile)
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(str(path), sess_options=opts, providers=providers)
    sess.disable_fallback()
    if device == "cuda" and sess.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError("CUDA_provider_fell_back_to_CPU")
    return sess


def finish_profile(sess, device):
    path = Path(sess.end_profiling())
    events = json.loads(path.read_text())
    counts = Counter(e.get("args", {}).get("provider") for e in events if e.get("args", {}).get("provider"))
    if device == "cuda" and not counts["CUDAExecutionProvider"]:
        raise RuntimeError("no_real_CUDA_kernel_in_profile")
    return {"profile": path.name, "kernel_events_by_provider": dict(counts)}


class React:
    def __init__(self, cache, device, output, cfg):
        self.device, self.cfg = device, cfg
        self.sess = session(cache / "react/yolo12m/react_pp_yolo12m.onnx", device, output / "react-profile")
        meta = self.sess.get_modelmeta().custom_metadata_map
        self.obj_names = dict(enumerate(json.loads(meta["obj_classes"]), 1))
        self.rel_names = dict(enumerate(json.loads(meta["rel_classes"])))
        spec = importlib.util.spec_from_file_location(
            "official_react_demo", cache / "SGG-Benchmark/demo/standalone_onnx_demo.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.preprocess = module.SGG_ONNX_Standalone.preprocess
        self.metadata = {"objects": self.obj_names, "relations": self.rel_names, "onnx_metadata": meta}

    def predict(self, path, native_path):
        import cv2

        frame = cv2.imread(str(path))
        tensor, ratio, pad = self.preprocess(self, frame)
        outputs = self.sess.run(None, {self.sess.get_inputs()[0].name: tensor})
        np.savez_compressed(native_path, **{o.name: v for o, v in zip(self.sess.get_outputs(), outputs, strict=False)})
        objects, relations, post = react_decode(
            outputs[0], outputs[1], self.obj_names, self.rel_names, ratio, pad, frame.shape[1], frame.shape[0], self.cfg
        )
        return objects, relations, {"raw_npz": native_path.name, "ratio": ratio, "pad": pad, **post}

    def close(self):
        return finish_profile(self.sess, self.device)
