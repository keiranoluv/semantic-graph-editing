"""Frozen offline models, one model at a time; per-case resumable metric evidence."""

import gc
import importlib.metadata
import time

import numpy as np
from common.paths import ROOT
from PIL import Image, ImageOps
from task3_vn100.evaluation.core import case_header, read_stage, save_stage, stage_binding
from task3_vn100.generation.core import atomic
from task3_vn100.image_graph import read, sha256

CACHE = ROOT / "data/cache/level3-task3-v1"


def check_assets():
    selected = [
        f
        for f in read(ROOT / "configs/level3/task3-assets-v1.json")["files"]
        if f["path"].startswith(("clip/", "lpips/", "torch/"))
    ]
    if not selected:
        raise ValueError("missing_metric_asset_lock")
    for f in selected:
        p = CACHE / f["path"]
        if not p.is_file() or p.stat().st_size != f["size"] or sha256(p) != f["sha256"]:
            raise ValueError("missing_or_changed_metric_checkpoint:" + str(p))
    return {f["path"]: f["sha256"] for f in selected}


def check_captions(records):
    from transformers import CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(CACHE / "clip", local_files_only=True)
    counts = []
    for r in records:
        text = "A photo depicts " + r["author"]["target_caption_en"]
        n = len(tokenizer(text, truncation=False)["input_ids"])
        if n > 77:
            raise ValueError("caption_exceeds_77_tokens:" + r["task"]["task_id"])
        counts.append(n)
    return {"captions": len(counts), "maximum_tokens": max(counts)}


def rgb(path):
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


class ClipScore:
    def __init__(self, device):
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        self.model = (
            CLIPModel.from_pretrained(CACHE / "clip", local_files_only=True).eval().requires_grad_(False).to(device)
        )
        self.processor = CLIPProcessor.from_pretrained(CACHE / "clip", local_files_only=True, use_fast=False)

    def score(self, record):
        import torch

        text = "A photo depicts " + record["author"]["target_caption_en"]
        image = rgb(record["image"]["target_path"])
        tokens = self.processor.tokenizer(text, truncation=False)
        if len(tokens["input_ids"]) > 77:
            raise ValueError("caption_exceeds_77_tokens")
        inp = self.processor(images=image, text=[text], padding=True, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self.model(**inp)
        cosine = float((out.image_embeds * out.text_embeds).sum(-1).item())
        return {"clipscore": 2.5 * max(0.0, cosine), "clip_cosine": cosine, "clip_text": text}


class LPIPS:
    def __init__(self, device):
        import lpips
        import torch

        self.device = device
        torch.hub.set_dir(str(CACHE / "torch"))
        self.model = (
            lpips.LPIPS(net="alex", version="0.1", model_path=str(CACHE / "lpips/alex.pth"), verbose=False)
            .eval()
            .requires_grad_(False)
            .to(device)
        )

    def distance(self, source, target):
        import torch

        source = source.resize(target.size, Image.Resampling.LANCZOS) if source.size != target.size else source

        def tensor(im):
            return (
                torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 127.5
                - 1
            )

        with torch.inference_mode():
            return float(self.model(tensor(source), tensor(target)).item())

    def score(self, record):
        source, target = rgb(record["image"]["source_path"]), rgb(record["image"]["target_path"])
        value = self.distance(source, target)
        return {
            "lpips": value,
            "negative_lpips": -value,
            "source_size": list(source.size),
            "evaluation_size": list(target.size),
        }


def validate_cache(pack, header, binding, device):
    if pack["case"] != header or pack["stage_binding"] != binding or pack["device"] != device or pack["status"] != "ok":
        raise ValueError("metric_resume_binding_mismatch")
    if not all(np.isfinite(v) for v in pack["values"].values() if isinstance(v, (float, int))):
        raise ValueError("nonfinite_cached_metric")


def run(out, cfg, records, device, smoke=False):
    import torch

    assets = check_assets()
    check_captions(records)
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.set_num_threads(4)
    if device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA_unavailable")
        if torch.cuda.get_device_capability()[0] >= 12 and "sm_120" not in torch.cuda.get_arch_list():
            raise ValueError("Torch_wheel_missing_RTX5090_kernels")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    selected = records[:1] if smoke else records
    stage = out / ("smoke-" + device if smoke else "metrics")
    binding = stage_binding(cfg, records)
    tracking = None
    metadata = {
        "device": device,
        "dtype": "float32",
        "batch_size": 1,
        "seed": cfg["seed"],
        "evaluation_eligible": not smoke,
        "assets": assets,
        "stage_binding": binding,
        "packages": {
            n: importlib.metadata.version(n) for n in ("torch", "torchvision", "transformers", "lpips", "Pillow")
        },
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
    }
    if (stage / "run.json").exists() and read(stage / "run.json") != metadata:
        raise ValueError("metric_runtime_changed_use_new_OUTPUT")
    atomic(stage / "run.json", metadata)
    if not smoke:
        import wandb

        tracking = wandb.init(
            project="level3-evaluation200-v1",
            name=out.name + "-metrics",
            dir=str(stage),
            config=metadata,
            mode="online",
        )
        atomic(stage / "wandb.json", {"id": tracking.id, "url": tracking.url})
    try:
        for name, factory in [("clip", ClipScore), ("lpips", LPIPS)]:
            missing = []
            for r in selected:
                p = stage / name / (r["task"]["task_id"] + ".json")
                if p.exists():
                    validate_cache(read_stage(p), case_header(r), binding, device)
                else:
                    missing.append(r)
            if not missing:
                continue
            backend = factory(device)
            for r in missing:
                if device == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                values = backend.score(r)
                if not all(np.isfinite(v) for v in values.values() if isinstance(v, (float, int))):
                    raise ValueError("nonfinite_metric")
                if device == "cuda":
                    torch.cuda.synchronize()
                memory = (
                    {
                        "allocated_bytes": torch.cuda.max_memory_allocated(),
                        "reserved_bytes": torch.cuda.max_memory_reserved(),
                    }
                    if device == "cuda"
                    else None
                )
                pack = {
                    "case": case_header(r),
                    "stage_binding": binding,
                    "status": "ok",
                    "device": device,
                    "values": values,
                    "seconds": time.perf_counter() - started,
                    "peak_vram": memory,
                }
                save_stage(stage / name / (r["task"]["task_id"] + ".json"), pack)
                if tracking:
                    tracking.log({name + "/" + k: v for k, v in values.items() if isinstance(v, (float, int))})
            if smoke and name == "lpips":
                im = rgb(selected[0]["image"]["target_path"])
                identical = backend.distance(im, im)
                if abs(identical) > 1e-7:
                    raise ValueError("LPIPS_identical_image_smoke_failed")
                atomic(stage / "sanity.json", {"identical_image_lpips": identical})
            del backend
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
        rows = []
        for r in selected:
            values = {}
            for name in ("clip", "lpips"):
                pack = read_stage(stage / name / (r["task"]["task_id"] + ".json"))
                validate_cache(pack, case_header(r), binding, device)
                values.update(pack["values"])
            rows.append({**case_header(r), **values})
        save_stage(stage / "results.json", {"evaluation_eligible": not smoke, "stage_binding": binding, "rows": rows})
        if tracking:
            tracking.finish()
        return {"output": str(stage), "cases": len(rows), "evaluation_eligible": not smoke}
    except Exception as exc:
        atomic(
            stage / "failure.json",
            {
                "error": repr(exc),
                "OOM": isinstance(exc, torch.OutOfMemoryError),
                "resume": "same command resumes completed per-case metric files",
            },
        )
        if tracking:
            tracking.finish(exit_code=1)
        raise
