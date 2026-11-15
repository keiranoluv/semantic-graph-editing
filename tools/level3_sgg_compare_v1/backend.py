"""Inference bridge to official SAM and ROBIN modules, without importing trainers."""

from __future__ import annotations

import importlib
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from tools.level3_image_graph_v1.core import digest, validate_box

from .adapter import validate_regions


def select_regions(native, size, config):
    width, height = size
    kept, excluded = [], []
    ranked = sorted(enumerate(native), key=lambda x: (-x[1]["area"], -x[1]["stability_score"], x[0]))
    for original_index, region in ranked:
        x, y, w, h = region["bbox"]
        box = [x, y, x + w, y + h]
        reason = None
        try:
            validate_box(box, width, height)
        except ValueError:
            reason = "invalid_box"
        if not config["min_area_fraction"] <= region["area"] / (width * height) <= config["max_area_fraction"]:
            reason = "area_filter"
        if reason is None and len(kept) >= config["max_regions"]:
            reason = "region_budget"
        if reason:
            excluded.append({"sam_index": original_index, "reason": reason})
            continue
        kept.append(
            {
                "region_id": len(kept) + 1,
                "sam_index": original_index,
                "bbox_xyxy": box,
                "segmentation": region["segmentation"],
                "area": region["area"],
                "predicted_iou": region["predicted_iou"],
                "stability_score": region["stability_score"],
                "native_region_sha256": digest(region),
                "origin": "sam_vit_h_automatic",
            }
        )
    if kept:
        validate_regions(kept, width, height)
    return kept, excluded


def load_sam(checkpoint, config, device):
    import torch
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    model = sam_model_registry[config["model_type"]](checkpoint=None)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    model.eval().requires_grad_(False).to(device)
    return SamAutomaticMaskGenerator(
        model,
        output_mode="coco_rle",
        **{
            k: config[k]
            for k in (
                "points_per_side",
                "points_per_batch",
                "pred_iou_thresh",
                "stability_score_thresh",
                "crop_n_layers",
            )
        },
    )


def propose_regions(generator, image, config):
    import torch

    with torch.inference_mode():
        native = generator.generate(np.asarray(image.convert("RGB")))
    # Convert numpy scalars and RLE bytes before immutable JSON serialization.
    import json

    native = json.loads(json.dumps(native, default=lambda x: x.decode("ascii") if isinstance(x, bytes) else x.item()))
    regions, excluded = select_regions(native, image.size, config)
    return {"native_regions": native, "regions": regions, "excluded": excluded}


def square_inputs(image, regions):
    """Pad RGB, binary masks and prompt boxes with the SAME offset; retain originals.

    Upstream demo pads RGB but leaves original masks/box coordinates unchanged.
    This explicit v1 correction is a preprocessing deviation, not a weight change.
    """
    from pycocotools import mask as mask_utils

    width, height = image.size
    validate_regions(regions, width, height)
    side = max(width, height)
    dx, dy = (side - width) // 2, (side - height) // 2
    canvas = Image.new("RGB", (side, side), tuple(int(v * 255) for v in (0.48145466, 0.4578275, 0.40821073)))
    canvas.paste(image.convert("RGB"), (dx, dy))
    masks, boxes = [], []
    for region in regions:
        encoded = dict(region["segmentation"])
        encoded["counts"] = encoded["counts"].encode("ascii")
        mask = mask_utils.decode(encoded)
        if mask.shape != (height, width) or not mask.any():
            raise ValueError("empty_or_wrong_size_mask")
        padded = np.zeros((side, side), dtype=np.uint8)
        padded[dy : dy + height, dx : dx + width] = mask
        masks.append(padded)
        x1, y1, x2, y2 = region["bbox_xyxy"]
        boxes.append([int(1000 * v / side) for v in (x1 + dx, y1 + dy, x2 + dx, y2 + dy)])
    return canvas, np.stack(masks), boxes, {"policy": "aligned_square_pad_v1", "offset_xy": [dx, dy], "side": side}


def make_prompt(boxes, question):
    refs = []
    for i, (x1, y1, x2, y2) in enumerate(boxes, 1):
        refs.append(f"region{i} <mask><pos> <|box_start|>({x1},{y1}),({x2},{y2})<|box_end|>")
    return (
        "<image>\nThis provides an overview of the picture.\n"
        f"There are {len(boxes)} regions in the image: " + ", ".join(refs) + ".\n " + question
    )


def import_upstream(repo):
    repo = Path(repo).resolve()
    if "svg" in sys.modules and Path(sys.modules["svg"].__file__).resolve().parent != repo / "svg":
        raise ValueError("svg_import_collision")
    sys.path.insert(0, str(repo))
    try:
        return (
            importlib.import_module("svg.model.language_model.robin_qwen"),
            importlib.import_module("svg.conversation"),
            importlib.import_module("svg.mm_utils"),
            importlib.import_module("svg.model.multimodal_encoder.clip_encoder"),
        )
    finally:
        sys.path.pop(0)


def restore_convnext_gamma(model, model_path, info):
    """Recover only the known Transformers 4.43 gamma->weight key rewrite.

    Read original safetensors directly; never accept arbitrary missing weights,
    change upstream code, or rewrite checkpoint files. Verify copied values.
    """
    import torch
    from safetensors import safe_open

    missing, unexpected = set(info["missing_keys"]), set(info["unexpected_keys"])
    if info["mismatched_keys"] or info["error_msgs"]:
        raise ValueError("robin_checkpoint_shape_or_load_error:" + str(info))
    pattern = r"model\.vision_tower\.vision_tower\.visual\.trunk\.stages\.\d+\.blocks\.\d+\.gamma"
    if not all(re.fullmatch(pattern, key) for key in missing) or unexpected != {
        key.removesuffix("gamma") + "weight" for key in missing
    }:
        raise ValueError("robin_unrecognized_checkpoint_keys:" + str(info))
    model_path = Path(model_path)
    index = json.loads((model_path / "model.safetensors.index.json").read_text())["weight_map"]
    parameters = dict(model.named_parameters())
    restored = []
    for key in sorted(missing):
        shard = (model_path / index[key]).resolve()
        if model_path.resolve() not in shard.parents:
            raise ValueError("unsafe_checkpoint_shard_path")
        with safe_open(shard, framework="pt", device="cpu") as tensors:
            original = tensors.get_tensor(key)
        target = parameters[key]
        if target.shape != original.shape or target.is_meta:
            raise ValueError("convnext_gamma_shape_or_device_mismatch")
        with torch.no_grad():
            target.copy_(original)
        if not torch.equal(target.detach().cpu(), original.to(target.dtype)):
            raise ValueError("convnext_gamma_restore_failed")
        restored.append({"key": key, "shard": index[key], "verified_exact_after_dtype_conversion": True})
    return restored


class RobinBackend:
    def __init__(self, repo, model_path, device="cuda"):
        import torch
        from transformers import AutoTokenizer

        model_path = Path(model_path).resolve()
        language, conversation, utils, vision = import_upstream(repo)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, use_fast=True, padding_side="right"
        )
        self.tokenizer.pad_token = self.tokenizer.pad_token or self.tokenizer.eos_token
        # <image> is a sentinel replaced by tokenizer_image_token(-200), not
        # necessarily a vocabulary entry. Mask/position tokens must exist.
        for token in ("<mask>", "<pos>"):
            if token not in self.tokenizer.get_vocab():
                raise ValueError("checkpoint_missing_multimodal_token")
        cfg = language.RobinQwenConfig.from_pretrained(model_path, local_files_only=True)
        original_vision_path = cfg.mm_vision_tower
        cfg.mm_vision_tower = str(model_path / "open_clip_pytorch_model.bin")
        if cfg.image_aspect_ratio != "pad" or cfg.tokenizer_model_max_length != 8192:
            raise ValueError("unexpected_robin_preprocessing_config")
        self.model, info = language.RobinQwenForCausalLM.from_pretrained(
            model_path,
            config=cfg,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            use_safetensors=True,
            attn_implementation="eager",
            output_loading_info=True,
        )
        restored = restore_convnext_gamma(self.model, model_path, info)
        self.model.eval().requires_grad_(False).to(device)
        for module in self.model.modules():
            module.tokenizer = self.tokenizer
        self.device = device
        self.processor = vision.get_image_processor(img_size=512)
        self.conv = conversation.conv_templates["qwen_2"]
        self.utils = utils
        self.loading_evidence = {
            "loading_info": info,
            "original_vision_path": original_vision_path,
            "restored_convnext_gamma": restored,
            "unresolved_loading_keys": [],
            "resolved_vision_path": cfg.mm_vision_tower,
            "config_modified_on_disk": False,
            "all_parameters_frozen": all(not p.requires_grad for p in self.model.parameters()),
            "attention": "eager",
            "dtype": "bfloat16",
        }

    def generate(self, image, regions, config, *, max_new_tokens=None):
        import torch
        from transformers import GenerationConfig

        canvas, masks, boxes, geometry = square_inputs(image, regions)
        user_prompt = make_prompt(boxes, config["question"])
        conv = self.conv.copy()
        conv.append_message(conv.roles[0], user_prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        ids = (
            self.utils.tokenizer_image_token(prompt, self.tokenizer, -200, return_tensors="pt")
            .unsqueeze(0)
            .to(self.device)
        )
        pixels = self.processor.preprocess(canvas, return_tensors="pt", do_center_crop=False)["pixel_values"].to(
            self.device, dtype=torch.bfloat16
        )
        masks = torch.from_numpy(masks).to(self.device, dtype=torch.bfloat16)
        maximum = max_new_tokens or config["max_new_tokens"]
        # ConvNeXt res4 at 512px is 32x32=1024 tokens, replacing one image token.
        effective_input = ids.shape[1] - 1 + 1024
        if effective_input + maximum > 8192:
            raise ValueError("robin_context_budget_exceeded; do not silently truncate regions")
        gen = GenerationConfig(
            do_sample=False,
            num_beams=1,
            max_new_tokens=maximum,
            repetition_penalty=config["repetition_penalty"],
            eos_token_id=[151645, 151643],
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
            return_dict_in_generate=True,
        )
        with torch.inference_mode():
            output = self.model.generate(
                ids, attention_mask=torch.ones_like(ids), images=pixels, masks=[masks], generation_config=gen
            )
        tokens = output.sequences[0].detach().cpu().tolist()
        text = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
        evidence = {
            "token_ids": tokens,
            "generated_tokens": len(tokens),
            "max_new_tokens": maximum,
            "hit_token_limit": len(tokens) >= maximum and tokens[-1] not in (151645, 151643),
            "prompt": prompt,
            "input_tokens": ids.shape[1],
            "effective_input_tokens": effective_input,
            "geometry": geometry,
            "do_sample": False,
            "semantic_confidence": None,
        }
        return text, evidence
