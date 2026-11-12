"""USER-RUN GPU inference v2; strict ID schema; immutable v1 code/cache preserved.

--review-only --plan is CPU-only. --full requires the actual 20-pair human CSV.
No edit instruction, source/target role or other image enters the model input.
"""

import argparse
import json
import re
from pathlib import Path

from PIL import Image

from src.knowledge.transe.core import sha256
from tools.infer_magicbrush_graphs import parse_model_graph, resize_for_model, select_images
from tools.prepare_magicbrush import DATA, ROOT, digest, encode, fixed
from tools.run_gqa_full import exclusive

PROMPT = """Describe visible objects in this ONE image as a compact scene graph.
Output ONLY a JSON object with exactly two keys: "nodes" and "edges".
Use up to 12 clearly visible object instances and up to 16 clear relations.

NODE FORMAT EXAMPLE ONLY (replace all semantic values and box coordinates with observations):
{"id":"n1","label":"visible singular noun","attributes":[],"bbox":[0,0,1,1]}
Every node has exactly id, label, attributes, bbox. Use unique string IDs n1, n2, ... .
Put color, material and clothing descriptions in attributes as short strings.
Prefer common stable nouns such as person, shirt or car; avoid guessing occupations or roles.
Use short atomic property strings, such as a single color/material, rather than long descriptions.
A bbox is ONE FLAT list of FOUR finite pixel numbers [x1,y1,x2,y2].
Do NOT output [[x1,y1,x2,y2]], bbox_2d, x/y/width/height or normalized coordinates.
Require 0 <= x1 < x2 <= image width and 0 <= y1 < y2 <= image height.
Choose tight boxes for visible instances, including distinguishable small objects.

EDGE FORMAT EXAMPLE ONLY:
{"subject":"n1","predicate":"visible relation","object":"n2"}
Both subject and object MUST be copied EXACTLY from id values in your nodes list.
Never use an object label, color, clothing description, attribute or undeclared ID as an endpoint.
If a relation requires an object you did not list with a visible box, omit THAT relation.
Do not express attributes as edges. Do not use self edges. edges may be [].
Use succinct relation names without an "is"/"are" prefix; keep names literal and consistent.

Before returning, check EVERY edge endpoint occurs in nodes and EVERY bbox is flat.
Use only claims supported by this image; never invent a desired modification or another image.
Do not add objects or relations from these format examples. No markdown, commentary or extra keys."""


def messages_for_image(width, height):
    text = PROMPT + f"\nActual model image frame: width={width}, height={height} pixels."
    return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]


def parse_model_graph_v2(response, frame_size):
    response = response.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", response, flags=re.DOTALL)
    raw = json.loads(fenced.group(1) if fenced else response)
    if not isinstance(raw, dict) or not isinstance(raw.get("nodes"), list):
        raise ValueError("Expected graph object with a node list")
    repairs = []
    for index, node in enumerate(raw["nodes"]):
        if not isinstance(node, dict):
            raise ValueError("Node must be an object")
        if "bbox" in node and "bbox_2d" in node:
            raise ValueError("Conflicting bbox keys; no guessing allowed")
        if "bbox" not in node and "bbox_2d" in node:
            node["bbox"] = node.pop("bbox_2d")
            repairs.append(f"nodes[{index}]: rename bbox_2d to bbox")
        box = node.get("bbox")
        if isinstance(box, list) and len(box) == 1 and isinstance(box[0], list) and len(box[0]) == 4:
            node["bbox"] = box[0]
            repairs.append(f"nodes[{index}]: unwrap one singleton bbox dimension")
    # Same strict geometry/schema/endpoint validator as v1. No boxes are clipped,
    # no nodes/edges are invented/deleted, no label-to-ID guesses are performed.
    graph = parse_model_graph(json.dumps(raw, ensure_ascii=False), frame_size)
    return graph, repairs


def run(args):
    config = json.loads(args.config.read_text())
    if args.full:
        from tools.magicbrush_review_gate_v2 import verify_gate

        gate = verify_gate(config, ROOT / config["review_csv"], ROOT / config["review_packet"], args.manifest)
        if not args.plan:
            fixed(ROOT / config["output"] / "full-review-gate.json", encode(gate))
    if (
        type(config["batch_size"]) is not int
        or not 1 <= config["batch_size"] <= 8
        or config["dtype"] not in ("bfloat16", "float16")
        or config["max_new_tokens"] <= 0
        or not 0 < config["min_pixels"] <= config["max_pixels"]
    ):
        raise ValueError("Invalid inference batch/dtype/token/image budget")
    manifest = json.loads(args.manifest.read_text())
    images = select_images(manifest, args.review_only)
    output = (ROOT / config["output"]).resolve()
    if not output.is_relative_to(ROOT / "data"):
        raise ValueError("Inference outputs must live under data/")
    if args.plan:
        print(
            json.dumps(
                {
                    "stage": "plan_only_no_GPU",
                    "scope": "pilot" if args.review_only else "full",
                    "images": len(images),
                    "pairs": 20 if args.review_only else len(manifest["records"]),
                    "model": config["model"],
                    "revision": config["revision"],
                    "batch_size": config["batch_size"],
                    "instructions_in_model_input": False,
                    "output": str(output),
                },
                indent=2,
            )
        )
        return
    import torch
    import transformers
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    if config["device"] != "cuda" or not torch.cuda.is_available():
        raise ValueError("User-run inference requires CUDA; use --plan for CPU preparation")
    identity = {
        "schema_version": 1,
        "dataset_manifest_sha256": sha256(args.manifest),
        "config": config,
        "prompt_sha256": digest(PROMPT),
        "model_inputs": "single_image_pixels_and_fixed_prompt_only",
        "format_policy": "lossless_bbox_key_or_singleton_unwrap_only; no endpoint guessing or edge deletion",
        "editing_instruction_supplied_to_model": False,
        "parameters_frozen": True,
        "torch_version": str(torch.__version__),
        "transformers_version": transformers.__version__,
        "source_sha256": {
            str(p.relative_to(ROOT)): sha256(p)
            for p in (
                Path(__file__),
                ROOT / "tools/infer_magicbrush_graphs.py",
                ROOT / "tools/magicbrush_graphs.py",
                ROOT / "tools/magicbrush_review_gate_v2.py",
                ROOT / "tools/edit_model_v1/data.py",
                ROOT / "tools/review_magicbrush.py",
            )
        },
    }
    fixed(output / "manifest.json", encode(identity))
    pending = []
    for image in images:
        path = ROOT / image["path"]
        if sha256(path) != image["sha256"]:
            raise ValueError("Image bytes changed")
        result = output / f"{image['sha256']}.json"
        if result.exists():
            previous = json.loads(result.read_text())
            if previous["image_sha256"] != image["sha256"] or previous["inference_identity_sha256"] != digest(identity):
                raise ValueError("Existing image prediction identity changed")
        else:
            pending.append(image)
    if not pending:
        print("[VERIFIED] all selected image predictions already exist")
        return
    torch.manual_seed(config["seed"])
    processor = AutoProcessor.from_pretrained(
        config["model"],
        revision=config["revision"],
        min_pixels=config["min_pixels"],
        max_pixels=config["max_pixels"],
        use_fast=False,
    )
    processor.tokenizer.padding_side = "left"
    model = (
        Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config["model"],
            revision=config["revision"],
            torch_dtype={"bfloat16": torch.bfloat16, "float16": torch.float16}[config["dtype"]],
            attn_implementation=config["attention"],
        )
        .to("cuda")
        .eval()
    )
    model.requires_grad_(False)
    tracker = None
    if args.wandb_mode != "disabled":
        import wandb

        tracker = wandb.init(
            entity="keiranoluv-hcmus",
            project="semantic-graph-editing",
            mode=args.wandb_mode,
            name=f"magicbrush-graphs-schema-v2-qwen25vl3b-bs{config['batch_size']}-{'pilot' if args.review_only else 'full'}",
            tags=[
                "magicbrush",
                "schema-v2",
                "frozen-inference",
                f"batch-{config['batch_size']}",
                "image-only",
                "pilot" if args.review_only else "full",
            ],
            config={
                **identity,
                "execution_scope": "pilot" if args.review_only else "full",
                "selected_images": len(images),
            },
        )
    try:
        failures = 0
        for start in range(0, len(pending), config["batch_size"]):
            batch = pending[start : start + config["batch_size"]]
            pictures = []
            for image in batch:
                with Image.open(ROOT / image["path"]) as opened:
                    pictures.append(resize_for_model(opened.convert("RGB"), config))
            texts = [
                processor.apply_chat_template(
                    messages_for_image(*picture.size), tokenize=False, add_generation_prompt=True
                )
                for picture in pictures
            ]
            inputs = processor(text=texts, images=pictures, do_resize=False, padding=True, return_tensors="pt").to(
                "cuda"
            )
            with torch.inference_mode():
                outputs = model.generate(**inputs, max_new_tokens=config["max_new_tokens"], do_sample=False)
            generated = outputs[:, inputs.input_ids.shape[1] :]
            responses = processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for image, response, picture in zip(batch, responses, pictures, strict=True):
                entry = {
                    "image_sha256": image["sha256"],
                    "inference_identity_sha256": digest(identity),
                    "label_origin": "model_prediction_not_ground_truth",
                    "raw_response": response,
                    "inference_frame_size": list(picture.size),
                    "coordinate_conversion": "absolute_model_frame_pixels_to_xyxy_1000",
                }
                try:
                    graph, normalizations = parse_model_graph_v2(response, picture.size)
                    entry.update(status="PASS", graph=graph, format_normalizations=normalizations)
                except (ValueError, TypeError, KeyError) as error:
                    entry.update(status="ERROR", error=str(error))
                    failures += 1
                fixed(output / f"{image['sha256']}.json", encode(entry))
            print(
                f"[INFERENCE] {min(start+len(batch),len(pending))}/{len(pending)} new images; invalid={failures}",
                flush=True,
            )
            if tracker:
                tracker.log(
                    {
                        "images_completed": min(start + len(batch), len(pending)),
                        "invalid_graphs": failures,
                        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                    }
                )
    finally:
        if tracker:
            tracker.finish()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/level2/magicbrush-graphs-qwen25vl3b-v2.json")
    parser.add_argument("--manifest", type=Path, default=DATA / "manifest.json")
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--review-only", action="store_true", help="Only the same fixed 20 train review pairs")
    scope.add_argument("--full", action="store_true", help="All 912 images; requires verified actual human reviews")
    parser.add_argument("--plan", action="store_true", help="CPU-only, never imports/loads the vision model")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    args = parser.parse_args(argv)
    if args.plan:
        run(args)
    else:
        config = json.loads(args.config.read_text())
        output = (ROOT / config["output"]).resolve()
        if not output.is_relative_to(ROOT / "data"):
            raise ValueError("Inference outputs must live under data/")
        with exclusive(ROOT / "runs/level2/.gqa-kg-training-device"), exclusive(output):
            run(args)


if __name__ == "__main__":
    main()
