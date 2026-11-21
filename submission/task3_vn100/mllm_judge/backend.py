"""Two full frames, in source/edited order, on one CUDA device."""

import time

from PIL import Image

from task3_vn100.mllm_judge.protocol import ROOT, checked, messages, snapshot


def frame(path, edge, multiple):
    with Image.open(path) as original:
        image = original.convert("RGB")
    native = image.size
    image.thumbnail((edge, edge), Image.Resampling.LANCZOS)
    content = image.size
    size = tuple(((n + multiple - 1) // multiple) * multiple for n in content)
    offset = tuple((n - c) // 2 for n, c in zip(size, content, strict=True))
    canvas = Image.new("RGB", size, (128, 128, 128))
    canvas.paste(image, offset)
    return canvas, {
        "native_wh": list(native),
        "content_wh": list(content),
        "offset_xy": list(offset),
        "canvas_wh": list(size),
    }


def encode(processor, template, cfg, case):
    images, sizes = zip(
        *(frame(ROOT / case[r], cfg["long_edge"], cfg["grid_multiple"]) for r in ("source", "target")), strict=True
    )
    prompt = processor.apply_chat_template(
        messages(template, case["instruction"]), tokenize=False, add_generation_prompt=True
    )
    batch = processor(text=[prompt], images=list(images), images_kwargs={"do_resize": False}, return_tensors="pt")
    patch = processor.image_processor.patch_size
    merge = processor.image_processor.merge_size
    grid = batch.image_grid_thw.tolist()
    if len(grid) != 2 or patch * merge != cfg["grid_multiple"]:
        raise ValueError("Unexpected Qwen vision grid")
    for geometry, (t, h, w) in zip(sizes, grid, strict=True):
        measured = [w * patch, h * patch]
        if t != 1 or measured != geometry["canvas_wh"] or max(measured) > cfg["long_edge"]:
            raise ValueError("Processor changed full-frame geometry")
        geometry["processor_wh"] = measured
    return batch, list(sizes)


def memory(torch):
    torch.cuda.synchronize()
    return {
        "allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }


class Judge:
    def __init__(self, cfg):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Required CUDA device with BF16 support")
        if "5090" not in torch.cuda.get_device_name(0) or "sm_120" not in torch.cuda.get_arch_list():
            raise RuntimeError("Expected RTX 5090 and PyTorch with sm_120 support")
        self.torch = torch
        self.cfg = cfg
        torch.manual_seed(cfg["seed"])
        self.template = checked(cfg["prompt"]).read_text()
        folder = snapshot(cfg)
        self.processor = AutoProcessor.from_pretrained(folder, local_files_only=True, use_fast=True)
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        self.model = (
            Qwen3VLForConditionalGeneration.from_pretrained(
                folder,
                dtype=torch.bfloat16,
                attn_implementation=cfg["attention"],
                local_files_only=True,
            )
            .eval()
            .requires_grad_(False)
            .to("cuda:0")
        )
        self.load = {
            "gpu": torch.cuda.get_device_name(0),
            "total_mib": torch.cuda.get_device_properties(0).total_memory / 2**20,
            "peak": memory(torch),
            "seconds": time.perf_counter() - start,
        }

    def __call__(self, case):
        torch = self.torch
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        batch, sizes = encode(self.processor, self.template, self.cfg, case)
        with torch.inference_mode():
            batch = batch.to("cuda:0")
            output = self.model.generate(
                **batch,
                do_sample=self.cfg["do_sample"],
                num_beams=self.cfg["num_beams"],
                max_new_tokens=self.cfg["max_new_tokens"],
            )
            new = output[0, batch.input_ids.shape[1] :]
            raw = self.processor.decode(new, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return {
            "raw": raw,
            "tokens": new.numel(),
            "input_tokens": batch.input_ids.shape[1],
            "frames": sizes,
            "peak": memory(torch),
            "seconds": time.perf_counter() - start,
        }
