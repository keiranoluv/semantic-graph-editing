"""Paired CLIP preprocessing experiment on the frozen NV3 development cohort."""

import argparse
import fcntl
import importlib.metadata
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import rankdata

from tools.level3_analysis_v1.core import cluster_indices, rho
from tools.level3_evaluation.core import CONFIG, ROOT, inputs, read, sha256
from tools.level3_evaluation.metrics import CACHE, ClipScore, check_captions, rgb
from tools.level3_qwen_unified.core import atomic

FROZEN = ROOT / "runs/level3/evaluation-v1/seed42/reports/0fc28934ad71812a/results.json"
FROZEN_HASH = "76b76a9418cd1578ef0cf37710c07cafe421cea66bdb16981517be5c70ef492d"
OUT = ROOT / "runs/level3/clip-crop-v1/seed42"
SIZE = 224
PAD_RGB = (123, 117, 104)  # CLIP channel means, rounded in 8-bit sRGB
BOOTSTRAPS = 2000
SEED = 42


def letterbox(image):
    scale = SIZE / max(image.size)
    w, h = (max(1, round(length * scale)) for length in image.size)
    canvas = Image.new("RGB", (SIZE, SIZE), PAD_RGB)
    canvas.paste(image.resize((w, h), Image.Resampling.BICUBIC), ((SIZE - w) // 2, (SIZE - h) // 2))
    return canvas


def cropped_roi_fraction(size, region):
    # Slow CLIPProcessor resizes the short side to 224, floors the long side,
    # then uses integer offsets for the center crop.
    w, h = (length * SIZE // min(size) for length in size)
    left, top = (w - SIZE) // 2, (h - SIZE) // 2
    x1, y1, x2, y2 = region
    width = max(0, min(x2, (left + SIZE) / w) - max(x1, left / w))
    height = max(0, min(y2, (top + SIZE) / h) - max(y1, top / h))
    return float(np.clip(1 - width * height / ((x2 - x1) * (y2 - y1)), 0, 1))


@torch.inference_mode()
def paired_scores(backend, image, caption):
    scores = []
    # The control goes through the frozen processor verbatim. At 224x224
    # its resize and crop are identities on the already letterboxed image.
    for view in (image, letterbox(image)):
        batch = backend.processor(images=view, text=[caption], padding=True, return_tensors="pt").to(backend.device)
        result = backend.model(**batch)
        cosine = (result.image_embeds * result.text_embeds).sum().item()
        if not np.isfinite(cosine):
            raise ValueError("nonfinite CLIP cosine")
        scores.append(2.5 * max(0, cosine))
    return scores


def correlations(rows, label):
    y = [r[label] for r in rows]
    return [rho([r[k] for r in rows], y) for k in ("center", "letterbox")]


def compare(rows, label):
    center, full = correlations(rows, label)
    deltas = []
    rng = np.random.default_rng(SEED)
    if center is not None and full is not None:
        for _ in range(BOOTSTRAPS):
            selected = [rows[i] for i in cluster_indices(rows, rng)]
            a, b = correlations(selected, label)
            if a is not None and b is not None:
                deltas.append(b - a)
    return {
        "pairs": len(rows),
        "images": len({r["image_id"] for r in rows}),
        "passes": sum(r[label] for r in rows),
        "center_rho": center,
        "letterbox_rho": full,
        "delta_rho": full - center if center is not None and full is not None else None,
        "paired_cluster_ci95": np.quantile(deltas, [0.025, 0.975]).tolist() if deltas else None,
        "valid_replicates": len(deltas),
    }


def run(device, smoke, dest, tracking=None):
    print("[1/3] Kiểm tra 200 cặp, nhãn tay, caption và checkpoint CLIP...", flush=True)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if sha256(CONFIG) != "19afdd88eda1ca1592b646d163d1a05b5f4260c8d0a163fcf42be637037d858a":
        raise ValueError("Frozen evaluation config changed")
    if sha256(FROZEN) != FROZEN_HASH:
        raise ValueError("Frozen human outcome or reference CLIPScore changed")
    config, records = inputs()
    reference = read(FROZEN)
    if [r["task"]["task_id"] for r in records] != [r["task_id"] for r in reference]:
        raise ValueError("Cohort or task order changed")
    check_captions(records)
    assets = {
        f["path"]: f["sha256"]
        for f in read(ROOT / "configs/level3/task3-assets-v1.json")["files"]
        if f["path"].startswith("clip/")
    }
    for name, expected in assets.items():
        if sha256(CACHE / name) != expected:
            raise ValueError("Frozen CLIP asset changed: " + name)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    backend = ClipScore(device)
    rows = []
    selected = [
        (r, ref)
        for r, ref in zip(records, reference, strict=True)
        if not smoke or r["author"]["index"] in (48, 58, 168)
    ]
    total = len(selected)
    print(f"[2/3] CLIP ViT-B/32 | {device} | FP32 | batch 1 | {total} cặp, hai cách xử lý/cặp", flush=True)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    for done, (record, ref) in enumerate(selected, 1):
        image = rgb(record["image"]["target_path"])
        caption = config["metric_protocol"]["clip_prefix"] + record["author"]["target_caption_en"]
        center, full = paired_scores(backend, image, caption)
        if abs(center - ref["clipscore"]) > 1e-4:
            raise ValueError(f"Frozen center score mismatch: {ref['task_id']}, {center}, {ref['clipscore']}")
        rows.append(
            {
                "index": record["author"]["index"],
                "task_id": ref["task_id"],
                "image_id": ref["image_id"],
                "overall_pass": ref["overall_pass"],
                "edit_fulfillment": ref["edit_fulfillment"],
                "graph_evaluable": ref["graph_consistency"] is not None,
                "center": center,
                "letterbox": full,
                "reference_center": ref["clipscore"],
                "delta_score": full - center,
                "target_size": list(image.size),
                "roi_cut": cropped_roi_fraction(image.size, record["author"]["grounding"]["region"]),
            }
        )
        if done == 1 or done % 10 == 0 or done == total:
            rate = done / (time.monotonic() - started)
            eta = (total - done) / rate
            peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None
            memory = "" if peak is None else f" | peak allocated {peak:.2f} GiB"
            print(f"  {done:3}/{total} | {rate:.2f} cặp/s | ETA inference {eta:.0f}s{memory}", flush=True)
            if tracking:
                tracking.log({"pairs_scored": done, "pairs_per_second": rate, "peak_VRAM_GiB": peak})
    for metric in ("center", "letterbox"):
        ranks = rankdata([r[metric] for r in rows])
        for row, rank in zip(rows, ranks, strict=True):
            row[metric + "_percentile"] = 100 * (rank - 1) / (len(rows) - 1)
    groups = {
        "all": rows,
        "edge_roi": [r for r in rows if r["roi_cut"] >= 0.25],
        "other_roi": [r for r in rows if r["roi_cut"] < 0.25],
        "graph_common": [r for r in rows if r["graph_evaluable"]],
    }
    subgroup_stats = {}
    if not smoke:
        print(f"[3/3] Spearman và bootstrap ghép theo ảnh nguồn ({BOOTSTRAPS} lượt/nhóm, CPU)", flush=True)
        for name, group in groups.items():
            subgroup_stats[name] = {label: compare(group, label) for label in ("overall_pass", "edit_fulfillment")}
            result = subgroup_stats[name]["overall_pass"]
            print(
                f"  {name} (n={len(group)}): center={result['center_rho']}, "
                f"full={result['letterbox_rho']}, delta CI95={result['paired_cluster_ci95']}",
                flush=True,
            )
            if tracking:
                tracking.summary[name] = subgroup_stats[name]
    summary = {
        "schema": "level3-clip-crop-v1",
        "smoke": smoke,
        "device": device,
        "seed": SEED,
        "batch": 1,
        "dtype": "float32",
        "assets": {k: v for k, v in assets.items() if k.startswith("clip/")},
        "metric_protocol": config["metric_protocol"],
        "bootstrap_replicates": BOOTSTRAPS,
        "config_sha256": sha256(CONFIG),
        "reference_sha256": FROZEN_HASH,
        "code_sha256": sha256(Path(__file__)),
        "packages": {k: importlib.metadata.version(k) for k in ("torch", "transformers", "Pillow", "scipy", "numpy")},
        "n": len(rows),
        "max_control_difference": max(abs(r["center"] - r["reference_center"]) for r in rows),
        "edge_definition": "at least 25% of the frozen instruction ROI area outside the target center crop",
        "subgroups": subgroup_stats,
        "wandb_url": tracking.url if tracking else None,
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
    }
    atomic(dest / "scores.json", rows)
    summary["scores_sha256"] = sha256(dest / "scores.json")
    atomic(dest / "summary.json", summary)
    print(f"{len(rows)} pairs: {dest / 'summary.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    dest = OUT / (args.device + ("-smoke" if args.smoke else ""))
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.device == "cuda" and not args.smoke:
            import wandb

            with wandb.init(
                project="level3-clip-crop-v1",
                name="clip-b32-crop-vs-letterbox-seed42",
                mode="online",
                dir=str(dest),
                config={
                    "batch_size": 1,
                    "seed": SEED,
                    "dtype": "float32",
                    "pairs": 200,
                    "reference_sha256": FROZEN_HASH,
                },
            ) as tracking:
                run(args.device, args.smoke, dest, tracking)
        else:
            run(args.device, args.smoke, dest)
