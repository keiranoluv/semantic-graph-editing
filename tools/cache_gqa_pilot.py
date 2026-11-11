"""Freeze a train/val cohort, then cache frozen CLIP crops directly from GQA ZIP."""

import argparse
import hashlib
import io
import json
import time
import zipfile
from collections import Counter
from pathlib import Path

import clip
import h5py
import numpy as np
import torch
from PIL import Image

from scripts.cache_vg150_regions import extract_vectors, scale_boxes_to_jpeg
from src.datasets.gqa import GQAAnnotationLoader, image_archive_index, normalize_gqa
from src.datasets.gqa_pilot import record_identity, select_cohort
from src.knowledge.transe.core import sha256
from tools.prepare_level2_gqa import write_fixed


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("data/raw/gqa"))
    p.add_argument("--manifest", type=Path, default=Path("data/processed/level2/gqa-v1/manifest.json"))
    p.add_argument("--cohort", type=Path, default=Path("reports/level2/gqa-pilot-v1-cohort.json"))
    p.add_argument("--output-dir", type=Path, default=Path("data/cache/gqa-pilot-clip-v1"))
    p.add_argument("--weights", type=Path, default=Path("data/cache/vg150-clip-vit-b32-v1/weights/ViT-B-32.pt"))
    p.add_argument("--train-size", type=int, default=512)
    p.add_argument("--val-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = p.parse_args(argv)
    if args.batch_size <= 0:
        p.error("batch-size must be positive")
    torch.set_num_threads(1)
    manifest = json.loads(args.manifest.read_text())
    cohort = select_cohort(manifest, sha256(args.manifest), args.train_size, args.val_size, args.seed)
    write_fixed(args.cohort, cohort)  # Pin before viewing any selected annotations or metrics.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    identity = {
        "schema_version": 1,
        "cohort_sha256": sha256(args.cohort),
        "manifest_sha256": sha256(args.manifest),
        "encoder": "openai.CLIP.ViT-B/32",
        "weights_sha256": sha256(args.weights),
        "preprocess": "default CLIP; GT xyxy scaled to JPEG, floor/ceil crop; L2 normalized",
        "visual_dim": 512,
        "dtype": "float16",
        "code_sha256": {
            s: sha256(Path(s))
            for s in (
                "tools/cache_gqa_pilot.py",
                "src/datasets/gqa.py",
                "src/datasets/gqa_pilot.py",
                "scripts/cache_vg150_regions.py",
            )
        },
    }
    write_fixed(args.output_dir / "identity.json", identity)
    final, partial = args.output_dir / "features.h5", args.output_dir / "features.h5.partial"
    if (args.output_dir / "manifest.json").exists():
        saved = json.loads((args.output_dir / "manifest.json").read_text())
        if any(saved[k] != v for k, v in identity.items()) or sha256(final) != saved["features_sha256"]:
            raise ValueError("Completed cache does not match identity/hash")
        print("[PASS] Existing completed cache verified", flush=True)
        return
    if final.exists():
        raise FileExistsError("Uncommitted final cache; inspect recovery before continuing")
    start = time.perf_counter()
    model, transform = clip.load(str(args.weights), device=args.device, jit=False)
    model.eval().requires_grad_(False)
    archive = args.root / "archives/images.zip"
    image_index = image_archive_index(archive)
    counts_by_split, done = {}, 0
    with zipfile.ZipFile(archive) as images, h5py.File(partial, "a") as cache:
        for split in ("train", "val"):
            loader = GQAAnnotationLoader(args.root, split, args.manifest)
            stats = Counter()
            for image_id in cohort["splits"][split]:
                record, counts = normalize_gqa(image_id, loader.annotations[image_id], loader.vocabulary)
                stats.update(counts)
                stats["candidate_pairs"] += len(record.objects) * (len(record.objects) - 1)
                stats["unique_gt_triplets"] += len(set(record.relations))
                if image_id in cache and cache[image_id].attrs.get("complete", False):
                    if cache[image_id].attrs["record_sha256"] != record_identity(record):
                        raise ValueError("Partial cache record mismatch")
                    stats["jpeg_size_mismatches"] += int(cache[image_id].attrs["jpeg_size_mismatch"])
                    continue
                if image_id in cache:
                    del cache[image_id]
                jpeg = images.read(image_index[image_id])
                with Image.open(io.BytesIO(jpeg)) as image:
                    image = image.convert("RGB")
                    mismatch = image.size != (record.width, record.height)
                    boxes = scale_boxes_to_jpeg(
                        [o.bbox_xyxy for o in record.objects], (record.width, record.height), image.size
                    )
                    vectors = extract_vectors(model, transform, image, boxes, args.batch_size, args.device)
                group = cache.create_group(image_id)
                group.create_dataset("features", data=vectors.astype(np.float16))
                group.attrs["record_sha256"] = record_identity(record)
                group.attrs["jpeg_sha256"] = hashlib.sha256(jpeg).hexdigest()
                group.attrs["jpeg_size_mismatch"] = mismatch
                group.attrs["complete"] = True
                stats["jpeg_size_mismatches"] += mismatch
                done += 1
                if done % 32 == 0:
                    cache.flush()
                    print(f"[CACHE] {done} new images; {time.perf_counter() - start:.1f}s", flush=True)
            counts_by_split[split] = dict(stats)
            del loader
    partial.replace(final)
    report = {
        **identity,
        "complete": True,
        "features_sha256": sha256(final),
        "statistics": counts_by_split,
        "wall_seconds_this_invocation": time.perf_counter() - start,
        "images_encoded_this_invocation": done,
        "feature_file_bytes": final.stat().st_size,
        "device": args.device,
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "torch_version": str(torch.__version__),
        "trainable_encoder_parameters": 0,
        "jpeg_provenance": "SHA256 of each decoded source JPEG stored in its HDF5 group",
    }
    write_fixed(args.output_dir / "manifest.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
