"""Official ONNX models, original preprocessing, explicit CUDA execution evidence."""

import gc
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from .adapter import react_decode


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


def detect_owl(cache, device, rows, locked, output, cfg):
    """Run detector stage then release it before loading the relation model."""
    import torch
    from torchvision.ops import nms
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    processor = Owlv2Processor.from_pretrained(cache / "owl", local_files_only=True, use_fast=False)
    model = Owlv2ForObjectDetection.from_pretrained(cache / "owl", local_files_only=True).eval().to(device)
    labels = cfg["object_vocabulary"]
    result = {}
    with torch.inference_mode():
        for row in rows:
            im = Image.open(locked / row["image_path"]).convert("RGB")
            inputs = processor(text=[[f"a photo of a {x}" for x in labels]], images=im, return_tensors="pt").to(device)
            raw = model(**inputs)
            np.savez_compressed(
                output / "native" / (row["image_id"] + "-owl.npz"),
                logits=raw.logits.cpu().numpy(),
                pred_boxes=raw.pred_boxes.cpu().numpy(),
                objectness_logits=raw.objectness_logits.cpu().numpy(),
            )
            pred = processor.post_process_object_detection(
                raw, threshold=cfg["box_threshold"], target_sizes=torch.tensor([im.size[::-1]])
            )[0]
            boxes, scores, classes = (pred[k].cpu() for k in ("boxes", "scores", "labels"))
            before_clip = boxes.clone()
            boxes[:, 0::2].clamp_(0, im.width)
            boxes[:, 1::2].clamp_(0, im.height)
            valid = ((boxes[:, 2:] - boxes[:, :2]) > 0).all(1)
            idx = torch.where(valid)[0]
            # Class-agnostic NMS prevents synonyms becoming duplicate instances.
            selected = idx[nms(boxes[idx], scores[idx], cfg["nms_iou"])[: cfg["max_boxes"]]]
            objects = [
                {
                    "label": labels[int(classes[i])],
                    "confidence": float(scores[i]),
                    "bbox_xyxy": boxes[i].tolist(),
                    "native_index": int(i),
                }
                for i in selected
            ]
            result[row["image_id"]] = {
                "objects": objects,
                "detector_native": {
                    "raw_npz": row["image_id"] + "-owl.npz",
                    "postprocess_boxes_before_clip": before_clip.tolist(),
                    "postprocess_scores": scores.tolist(),
                    "postprocess_labels": classes.tolist(),
                    "selected_indices": selected.tolist(),
                    "device": str(model.device),
                },
            }
    del raw, inputs, model, processor
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


class Relate:
    def __init__(self, cache, device, output, cfg):
        sys.path.insert(0, str(cache / "RelateAnything"))
        from deploy.postprocess import ThresholdConfig, decode
        from deploy.runtime import OnnxRelationHead

        # Use the official preprocessing/vocabulary/decoding with our profiled session.
        class ProfiledHead(OnnxRelationHead):
            def _build_engine(head, path, threads, providers):
                head.sess = session(path, device, output / "relate-profile")
                head.input_names = [x.name for x in head.sess.get_inputs()]
                return [x.name for x in head.sess.get_outputs()]

        self.head = ProfiledHead(str(cache / "relate/relateanything.onnx"), str(cache / "relate/predicate_bank.npz"))
        if not self.head.emits_logits:
            raise ValueError("require_raw_logit_ONNX_export")
        self.head.set_predicates(self.head.available_predicates())
        if cfg["max_boxes"] > self.head.max_boxes:
            raise ValueError("box_budget_exceeds_export")
        self.cfg = ThresholdConfig(
            threshold=cfg["relation_threshold"],
            topk=cfg["topk"],
            calib_a=self.head.contract.calib_a,
            calib_b=self.head.contract.calib_b,
            pair_weight=1.0,
            box_score_weight=True,
            max_per_pair=1,
        )
        self.decode, self.device = decode, device
        self.metadata = {
            "predicates": self.head.predicates,
            "vocabulary_scope": "entire_released_ONNX_predicate_bank",
            "calibration": self.cfg.__dict__,
            "max_boxes": self.head.max_boxes,
        }

    def predict(self, path, objects, native_path):
        import cv2

        if len(objects) < 2:
            return [], {"relation_stage": "skipped_fewer_than_two_detections"}
        boxes = np.array([o["bbox_xyxy"] for o in objects], dtype=np.float32)
        scores = np.array([o["confidence"] for o in objects], dtype=np.float32)
        raw = self.head(cv2.imread(str(path)), boxes)
        np.savez_compressed(
            native_path,
            **dict(zip(("pred_logits", "pair_logits", "sub_idx", "obj_idx", "valid_mask"), raw, strict=False)),
        )
        trips = self.decode(*raw, self.head.predicates, self.cfg, boxes_xyxy=boxes, box_scores=scores)
        return [
            {
                "subject": t.subject_idx,
                "object": t.object_idx,
                "predicate": t.predicate,
                "confidence": t.score,
                "confidence_kind": "official_calibrated_relation_score",
                "ranking_score": t.score * float(scores[t.subject_idx]) * float(scores[t.object_idx]),
            }
            for t in trips
        ], {"raw_npz": native_path.name}

    def close(self):
        return finish_profile(self.head.sess, self.device)
