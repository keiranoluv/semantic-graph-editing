"""Frozen model adapters. Called only by the user's explicit inference command.

Each stage releases its model before the next stage. No model generates graphs
or operations. Detected boxes and raw query scores are retained as evidence.
"""

import gc

import numpy as np
from PIL import Image

from tools.magicbrush_graphs import iou
from tools.magicbrush_grounded_v1.language import COLORS
from tools.magicbrush_grounded_v1.policy import cosine
from tools.prepare_magicbrush import ROOT


def release(model):
    import torch

    del model
    gc.collect()
    torch.cuda.empty_cache()


def caption_images(images, config):
    import torch
    from transformers import BlipForConditionalGeneration, BlipProcessor

    processor = BlipProcessor.from_pretrained(
        config["caption_model"], revision=config["caption_revision"], use_fast=False
    )
    model = BlipForConditionalGeneration.from_pretrained(config["caption_model"], revision=config["caption_revision"])
    model.to("cuda").eval().requires_grad_(False)
    result = []
    with torch.inference_mode():
        for start in range(0, len(images), config["batch_size"]):
            inputs = processor(images=images[start : start + config["batch_size"]], return_tensors="pt").to("cuda")
            tokens = model.generate(
                **inputs, max_new_tokens=config["caption_max_new_tokens"], do_sample=False, num_beams=1
            )
            result.extend(processor.batch_decode(tokens, skip_special_tokens=True))
    model.to("cpu")
    release(model)
    return result


def query_text(queries):
    text, spans = "", []
    for query in queries:
        start = len(text)
        text += query + ". "
        spans.append((start, start + len(query)))
    return text.strip(), spans


def decode_detections(logits, raw_boxes, token_offsets, spans, queries, focus, config):
    """Mean sigmoid over all subword tokens of the exact query, pre-NMS max.

    Unlike free-text postprocessing, class strings always come from the input
    query vocabulary. A query with missing/truncated tokens is an error.
    """
    logits, raw_boxes = np.asarray(logits), np.asarray(raw_boxes)
    # Transformers pads unused text positions with -inf by design. Only actual
    # query tokens must be finite; NaN and +inf are invalid everywhere.
    if (
        logits.ndim != 2
        or raw_boxes.shape != (logits.shape[0], 4)
        or np.isnan(logits).any()
        or np.isposinf(logits).any()
        or not np.isfinite(raw_boxes).all()
    ):
        raise ValueError("Malformed detector heads")
    if np.any(raw_boxes < 0) or np.any(raw_boxes > 1):
        raise ValueError("Detector normalized cxcywh outside expected range")
    probabilities = 1 / (1 + np.exp(-np.clip(logits, -80, 80)))
    boxes = np.concatenate((raw_boxes[:, :2] - raw_boxes[:, 2:] / 2, raw_boxes[:, :2] + raw_boxes[:, 2:] / 2), axis=1)
    clipped = np.clip(boxes, 0, 1) * 1000
    candidates, presence = [], {}
    for query, (start, end) in zip(queries, spans, strict=True):
        tokens = [i for i, (a, b) in enumerate(token_offsets) if b > a and a < end and b > start]
        if not tokens or max(tokens) >= logits.shape[1] or max(token_offsets[i][1] for i in tokens) < end:
            raise ValueError("Grounding query missing or truncated")
        if not np.isfinite(logits[:, tokens]).all():
            raise ValueError("Non-finite real query token scores")
        scores = probabilities[:, tokens].mean(axis=1)
        presence[query] = float(scores.max())
        keep = []
        for index in sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i)):
            box = clipped[index].tolist()
            if scores[index] < config["detector_min_score"] or box[0] >= box[2] or box[1] >= box[3]:
                continue
            if any(iou(box, other["bbox"]) > config["nms_iou"] for other in keep):
                continue
            keep.append(
                {
                    "label": query,
                    "bbox": box,
                    "detector_score": float(scores[index]),
                    "detector_query_index": index,
                    "frame_clipped": bool(np.any(boxes[index] < 0) or np.any(boxes[index] > 1)),
                }
            )
        candidates.extend(keep)
    # Source graphs are partial. An instruction focus takes precedence over an
    # overlapping caption noun; competing focus names instead cause abstention.
    ordered = sorted(candidates, key=lambda n: (n["label"] not in focus, -n["detector_score"], n["label"], n["bbox"]))
    selected, ambiguous = [], False
    for node in ordered:
        overlapping = [n for n in selected if n["label"] != node["label"] and iou(n["bbox"], node["bbox"]) > 0.85]
        if overlapping:
            if node["label"] in focus and any(n["label"] in focus for n in overlapping):
                ambiguous = True
            else:
                continue
        selected.append(node)
    overflow = len(selected) > config["max_nodes"]
    selected = sorted(selected[: config["max_nodes"]], key=lambda n: (n["bbox"], n["label"]))
    for i, node in enumerate(selected):
        node["id"] = f"d{i+1:03d}"
    return {
        "nodes": selected,
        "presence_scores": presence,
        "overflow": overflow,
        "ambiguous_focus": ambiguous,
        "all_thresholded_candidates": candidates,
        "query_score_policy": "mean_sigmoid_over_exact_query_tokens; maximum_all_900_boxes_before_NMS",
    }


def detect_images(images, vocabularies, foci, config):
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        config["detector_model"], revision=config["detector_revision"], use_fast=False
    )
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        config["detector_model"],
        revision=config["detector_revision"],
        disable_custom_kernels=config["detector_disable_custom_kernels"],
    )
    model.to("cuda").eval().requires_grad_(False)
    result = []
    with torch.inference_mode():
        for start in range(0, len(images), config["batch_size"]):
            vocab = vocabularies[start : start + config["batch_size"]]
            prompts = [query_text(q) for q in vocab]
            inputs = processor(
                images=images[start : start + config["batch_size"]],
                text=[p[0] for p in prompts],
                padding=True,
                return_tensors="pt",
            )
            outputs = model(**inputs.to("cuda"))
            for k, (text, spans) in enumerate(prompts):
                offsets = processor.tokenizer(text, return_offsets_mapping=True)["offset_mapping"]
                result.append(
                    decode_detections(
                        outputs.logits[k].float().cpu().numpy(),
                        outputs.pred_boxes[k].float().cpu().numpy(),
                        offsets,
                        spans,
                        vocab[k],
                        foci[start + k],
                        config,
                    )
                )
    model.to("cpu")
    release(model)
    return result


def crop(image, box):
    w, h = image.size
    a, b, c, d = box
    return image.crop(
        (
            int(a * w / 1000),
            int(b * h / 1000),
            max(int(a * w / 1000) + 1, int(np.ceil(c * w / 1000))),
            max(int(b * h / 1000) + 1, int(np.ceil(d * h / 1000))),
        )
    )


def color_fractions(image):
    hsv = np.asarray(image.convert("HSV"), dtype=float) / 255
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    masks = {
        "black": v < 0.2,
        "white": (s < 0.15) & (v > 0.8),
        "gray": (s < 0.15) & (v >= 0.2) & (v <= 0.8),
        "red": ((h < 0.035) | (h > 0.95)) & (s >= 0.3) & (v >= 0.3),
        "orange": (h >= 0.035) & (h < 0.1) & (s >= 0.3) & (v >= 0.65),
        "yellow": (h >= 0.1) & (h < 0.19) & (s >= 0.3) & (v >= 0.3),
        "green": (h >= 0.19) & (h < 0.46) & (s >= 0.3) & (v >= 0.3),
        "blue": (h >= 0.46) & (h < 0.72) & (s >= 0.3) & (v >= 0.3),
        "purple": (h >= 0.72) & (h < 0.84) & (s >= 0.3) & (v >= 0.3),
        "pink": (h >= 0.84) & (h <= 0.95) & (s >= 0.15) & (v >= 0.6),
        "brown": (h >= 0.035) & (h < 0.1) & (s >= 0.3) & (v >= 0.2) & (v < 0.65),
    }
    return {name: float(mask.mean()) for name, mask in masks.items()}


def verify_pairs(pairs, proposals, config):
    import clip
    import torch

    model, preprocess = clip.load(str(ROOT / config["clip_weights"]), device="cuda", jit=False)
    model.float().eval().requires_grad_(False)
    texts = sorted(
        {f"a photo of a {node['label']}" for proposal in proposals for node in proposal["nodes"]}
        | {f"a photo of a {c} {node['label']}" for proposal in proposals for node in proposal["nodes"] for c in COLORS}
    )
    text_features = {}
    with torch.inference_mode():
        for start in range(0, len(texts), config["clip_batch_size"]):
            batch = texts[start : start + config["clip_batch_size"]]
            features = model.encode_text(clip.tokenize(batch, truncate=False).to("cuda")).float()
            features = torch.nn.functional.normalize(features, dim=-1).cpu().numpy()
            text_features.update(zip(batch, features, strict=True))
    evidence = []
    for index, (source, target) in enumerate(pairs):
        size = config["comparison_size"]
        # MagicBrush often stores original 500px -> edited 1024px. Compare in an
        # explicitly declared normalized frame; never pretend pixels were aligned.
        frames = [im.resize((size, size), Image.Resampling.BILINEAR) for im in (source, target)]
        arrays = [np.asarray(im, dtype=np.float32) / 255 for im in frames]
        mask = np.abs(arrays[0] - arrays[1]).mean(axis=-1) > config["pixel_change_threshold"]
        record = {
            "source_anchors": {},
            "target_anchors": {},
            "change_mask": mask,
            "changed_pixels": int(mask.sum()),
            "comparable_aspect_ratio": abs(source.width / source.height - target.width / target.height) < 0.01,
            "comparison_transform": f"both images resized bilinear to {size}x{size}; normalized xyxy; no optical alignment",
        }
        jobs, crops = [], []
        for role_index, role in enumerate(("source", "target")):
            for node in proposals[2 * index + role_index]["nodes"]:
                a, b = [crop(im, node["bbox"]) for im in frames]
                jobs.append((role, role_index, node, a, b))
                crops.extend((a, b))
        features = []
        with torch.inference_mode():
            for start in range(0, len(crops), config["clip_batch_size"]):
                tensor = torch.stack([preprocess(im) for im in crops[start : start + config["clip_batch_size"]]]).to(
                    "cuda"
                )
                vector = model.encode_image(tensor).float()
                features.extend(torch.nn.functional.normalize(vector, dim=-1).cpu().numpy())
        for k, (role, role_index, node, a, b) in enumerate(jobs):
            fa, fb = features[2 * k : 2 * k + 2]
            own = (fa, fb)[role_index]
            text = text_features[f"a photo of a {node['label']}"]
            proof = {
                "cosine": cosine(fa, fb),
                "mae": float(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)).mean() / 255),
                "source_class_score": float(fa @ text),
                "target_class_score": float(fb @ text),
            }
            record[role + "_anchors"][node["id"]] = proof
            node["embedding"] = own.tolist()
            node["color_scores"] = {c: float(own @ text_features[f"a photo of a {c} {node['label']}"]) for c in COLORS}
            node["color_fractions"] = color_fractions((a, b)[role_index])
        evidence.append(record)
    model.to("cpu")
    release(model)
    return evidence
