"""USER-RUN GPU inference: frozen VLM sees each image independently, never edits.

--plan is CPU-only. Actual inference must only be run by the user after commit.
The source/target role and instruction are not included in model messages.
"""

import argparse
import json
import math
import re
from pathlib import Path

from PIL import Image

from src.knowledge.transe.core import sha256
from tools.magicbrush_graphs import validate_graph
from tools.prepare_magicbrush import DATA, ROOT, digest, encode, fixed
from tools.run_gqa_full import exclusive

PROMPT = """Describe only what is visibly present in this single image as a compact scene graph.
Return one JSON object with exactly two keys: "nodes" and "edges". No markdown or explanation.
Include at most 12 visually important object instances and 16 clear relationships.
Each node must contain exactly "id" (unique string), "label" (short English singular noun),
"attributes" (list of visible properties such as color/material, never inferred intentions),
and "bbox" ([x1,y1,x2,y2] in absolute pixels of the supplied image frame).
Use positive-area boxes that tightly enclose the visible instance. Distinguish repeated objects.
Each edge must contain exactly "subject" (node id), "predicate" (short English relation),
and "object" (a different node id). Avoid uncertain objects, properties and relationships.
Do not describe any imagined modification or desired state. All claims must refer to this image.
Use empty lists only when the image genuinely contains no identifiable objects or relationships."""


def select_images(manifest, review_only=False):
    selected = set(manifest["review_ids"]) if review_only else {r["id"] for r in manifest["records"]}
    images = {}
    for row in manifest["records"]:
        if row["id"] in selected:
            for role in ("source_image", "target_image"):
                item = row[role]
                if item["sha256"] in images and images[item["sha256"]] != item:
                    raise ValueError("Conflicting metadata for the same image bytes")
                images[item["sha256"]] = item
    return [images[key] for key in sorted(images)]


def messages_for_image(width, height):
    prompt = (
        PROMPT
        + f"\nImage frame: width={width}, height={height} pixels. Box bounds: x in [0,{width}], y in [0,{height}]."
    )
    return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]


def parse_model_graph(response, frame_size):
    """Convert declared model-frame pixel coordinates; never guess or clip boxes."""
    response = response.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", response, flags=re.DOTALL)
    raw = json.loads(fenced.group(1) if fenced else response)
    width, height = frame_size
    if width <= 0 or height <= 0:
        raise ValueError("Invalid model frame")
    for node in raw["nodes"]:
        box = node["bbox"]
        if (
            not isinstance(box, list)
            or len(box) != 4
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in box)
        ):
            raise ValueError("Invalid pixel box")
        node["bbox"] = [v * 1000 / size for v, size in zip(box, (width, height, width, height), strict=True)]
    return validate_graph(raw)


def resize_for_model(image, config):
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

    height, width = smart_resize(
        image.height, image.width, factor=28, min_pixels=config["min_pixels"], max_pixels=config["max_pixels"]
    )
    return image.resize((width, height), Image.Resampling.BICUBIC)


def run(args):
    config = json.loads(args.config.read_text())
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
        "editing_instruction_supplied_to_model": False,
        "parameters_frozen": True,
        "torch_version": str(torch.__version__),
        "transformers_version": transformers.__version__,
        "source_sha256": {
            str(p.relative_to(ROOT)): sha256(p) for p in (Path(__file__), ROOT / "tools/magicbrush_graphs.py")
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
            name=f"magicbrush-graphs-qwen25vl3b-bs{config['batch_size']}",
            tags=["magicbrush", "frozen-inference", f"batch-{config['batch_size']}", "image-only"],
            config=identity,
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
                    entry.update(status="PASS", graph=parse_model_graph(response, picture.size))
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
    parser.add_argument("--config", type=Path, default=ROOT / "configs/level2/magicbrush-graphs-qwen25vl3b-v1.json")
    parser.add_argument("--manifest", type=Path, default=DATA / "manifest.json")
    parser.add_argument("--review-only", action="store_true", help="Only the fixed 20 train review pairs")
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
