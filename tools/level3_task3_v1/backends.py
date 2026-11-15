"""Sequential frozen IP2P, REACT++, CLIP and LPIPS stages."""

import gc
import hashlib

import numpy as np
from PIL import Image

from tools.level3_image_graph_v1.core import sha256
from tools.level3_sgg_pretrained_v2.adapter import adapt
from tools.level3_sgg_pretrained_v2.assets import CACHE as SGG_CACHE
from tools.level3_sgg_pretrained_v2.backends import React

from .core import CACHE


def image_seed(seed, task_id):
    return int.from_bytes(hashlib.sha256(f"{seed}|{task_id}".encode()).digest()[:4], "big")


def resize_source(path, max_side):
    with Image.open(path) as im:
        im = im.convert("RGB")
        scale = min(1, max_side / max(im.size))
        size = tuple(max(64, int(round(d * scale / 8)) * 8) for d in im.size)
        return im.resize(size, Image.Resampling.LANCZOS)


def release():
    import torch

    gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()


class Editor:
    def __init__(self, device, cfg):
        import torch
        from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionInstructPix2PixPipeline

        self.device = device
        self.cfg = cfg
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            CACHE / "editor",
            variant="fp16",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            local_files_only=True,
            use_safetensors=True,
        )
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(device)
        self.pipe.set_progress_bar_config(disable=True)
        for component in (self.pipe.unet, self.pipe.vae, self.pipe.text_encoder, self.pipe.safety_checker):
            if component is not None:
                component.eval().requires_grad_(False)
        self.pipe.enable_attention_slicing()

    def edit(self, source, prompt, seed):
        import torch

        with torch.inference_mode():
            out = self.pipe(
                prompt=prompt,
                image=source,
                num_inference_steps=self.cfg["steps"],
                guidance_scale=self.cfg["guidance_scale"],
                image_guidance_scale=self.cfg["image_guidance_scale"],
                generator=torch.Generator(device=self.device).manual_seed(seed),
            )
        flagged = bool(out.nsfw_content_detected and out.nsfw_content_detected[0])
        return out.images[0], {
            "safety_checker_flagged": flagged,
            "actual_seed": seed,
            "steps": self.cfg["steps"],
            "scheduler": type(self.pipe.scheduler).__name__,
        }


class Extractor:
    def __init__(self, device, output, react_config):
        self.backend = React(SGG_CACHE, device, output, react_config)

    def extract(self, path, identifier, native_path, provenance):
        with Image.open(path) as im:
            w, h = im.size
        row = {"image_id": identifier, "sha256": sha256(path), "width": w, "height": h, "image_path": str(path)}
        objects, relations, native = self.backend.predict(path, native_path)
        return adapt(row, "react++", objects, relations, provenance, native)

    def close(self):
        return self.backend.close()


class ClipScore:
    def __init__(self, device):
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        self.model = (
            CLIPModel.from_pretrained(CACHE / "clip", local_files_only=True).eval().requires_grad_(False).to(device)
        )
        self.processor = CLIPProcessor.from_pretrained(CACHE / "clip", local_files_only=True, use_fast=False)

    def score(self, image, caption):
        import torch

        # Reference-free CLIPScore: 2.5 max(cos(image, prefixed target caption), 0).
        text = "A photo depicts " + caption
        tokens = self.processor.tokenizer(text, return_tensors="pt", truncation=False)
        if tokens["input_ids"].shape[1] > self.model.config.text_config.max_position_embeddings:
            raise ValueError("target_caption_exceeds_CLIP_token_limit")
        inp = self.processor(images=image, text=[text], return_tensors="pt", padding=True).to(self.device)
        with torch.inference_mode():
            out = self.model(**inp)
            cosine = float((out.image_embeds * out.text_embeds).sum(dim=-1)[0])
        return {"clipscore": 2.5 * max(cosine, 0), "clip_cosine": cosine, "clip_text": text}


class LPIPS:
    def __init__(self, device):
        import lpips
        import torch

        torch.hub.set_dir(str(CACHE / "torch"))  # verified AlexNet checkpoint is already here
        self.device = device
        self.model = (
            lpips.LPIPS(net="alex", version="0.1", model_path=str(CACHE / "lpips/alex.pth"), verbose=False)
            .eval()
            .requires_grad_(False)
            .to(device)
        )

    def score(self, source, target):
        import torch

        if source.size != target.size:
            raise ValueError("LPIPS_requires_same_resolution")

        def tensor(im):
            return (
                torch.from_numpy(np.asarray(im.convert("RGB")).copy())
                .permute(2, 0, 1)
                .float()
                .unsqueeze(0)
                .to(self.device)
                / 127.5
                - 1
            )

        with torch.inference_mode():
            value = float(self.model(tensor(source), tensor(target)).item())
        return {"lpips": value, "negative_lpips": -value}
