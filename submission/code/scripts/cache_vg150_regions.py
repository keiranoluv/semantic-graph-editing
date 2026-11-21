"""Cache frozen OpenAI CLIP ViT-B/32 vectors for every pinned VG150 GT box."""

import argparse
import json
import math
from pathlib import Path

import clip
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.datasets.vg150 import VG150AnnotationLoader
from src.knowledge.transe.core import sha256

SOURCE_FILES = ("VG-SGG-with-attri.h5", "image_data.json", "splits.json", "images_manifest.json")


def cache_identity(root: Path, weights_path: Path, box_policy: str) -> dict:
    return {
        "schema_version": 1,
        "encoder": "openai.CLIP.ViT-B/32",
        "weights": "openai_pretrained_ViT-B-32",
        "weights_sha256": sha256(weights_path),
        "preprocess": "clip.load ViT-B/32 default; crop=floor/ceil GT xyxy scaled to JPEG dimensions; L2 normalized",
        "box_policy": box_policy,
        "visual_dim": 512,
        "storage_dtype": "float16",
        "source_sha256": {name: sha256(root / name) for name in SOURCE_FILES if (root / name).is_file()},
    }


def extract_vectors(model, transform, image: Image.Image, boxes: list[tuple], batch_size: int, device) -> np.ndarray:
    batches = []
    for start in range(0, len(boxes), batch_size):
        crops = []
        for x1, y1, x2, y2 in boxes[start : start + batch_size]:
            bounds = (math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2))
            crops.append(transform(image.crop(bounds)))
        with torch.inference_mode():
            vectors = model.encode_image(torch.stack(crops).to(device)).float()
            vectors = F.normalize(vectors, dim=-1).cpu().numpy()
        if vectors.shape[1:] != (512,) or not np.isfinite(vectors).all():
            raise ValueError("Encoder returned invalid region vectors")
        batches.append(vectors)
    return np.concatenate(batches, axis=0) if batches else np.empty((0, 512), dtype=np.float32)


def scale_boxes_to_jpeg(
    boxes: list[tuple], annotation_size: tuple[int, int], jpeg_size: tuple[int, int]
) -> list[tuple]:
    """Map boxes from aligned annotation coordinates to source JPEG pixels."""
    aw, ah = annotation_size
    jw, jh = jpeg_size
    if min(aw, ah, jw, jh) <= 0:
        raise ValueError("Image dimensions must be positive")
    sx, sy = jw / aw, jh / ah
    return [(x1 * sx, y1 * sy, x2 * sx, y2 * sy) for x1, y1, x2, y2 in boxes]


def initialize_cache(path: Path, root: Path, identity: dict) -> None:
    with h5py.File(root / "VG-SGG-with-attri.h5") as source:
        first = source["img_to_first_box"][:].astype(np.int64)
        last = source["img_to_last_box"][:].astype(np.int64)
        total_boxes = len(source["boxes_1024"])
    metadata = json.loads((root / "image_data.json").read_text(encoding="utf-8"))
    with h5py.File(path, "x") as cache:
        cache.attrs["identity"] = json.dumps(identity, sort_keys=True)
        cache.create_dataset("features", (total_boxes, 512), dtype="float16")
        cache.create_dataset("boxes_xyxy", (total_boxes, 4), dtype="float32")
        cache.create_dataset("row_first", data=first)
        cache.create_dataset("row_last", data=last)
        cache.create_dataset("image_id", data=np.asarray([int(item["image_id"]) for item in metadata], dtype=np.int64))
        cache.create_dataset("done", (len(metadata),), dtype="bool", fillvalue=False)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/vg150"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/cache/vg150-clip-vit-b32-v1"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--box-policy", choices=("strict", "clip"), default="clip")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--existing-only", action="store_true", help="CPU smoke: skip images not yet downloaded")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or (args.max_images is not None and args.max_images <= 0):
        parser.error("Batch size and max-images must be positive")
    splits = json.loads((args.root / "splits.json").read_text(encoding="utf-8"))
    selected_rows = set(splits["train"]) | set(splits["val"]) | set(splits["test"])
    print(f"[PLAN] {len(selected_rows)} images, frozen CLIP ViT-B/32, device={args.device}", flush=True)
    if args.dry_run:
        return
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if not args.existing_only:
        images_manifest = json.loads((args.root / "images_manifest.json").read_text(encoding="utf-8"))
        if len(images_manifest["images"]) != len(selected_rows):
            raise ValueError("Complete JPEG manifest is required for full cache extraction")
        image_hashes = {int(item["row"]): item["sha256"] for item in images_manifest["images"]}
    else:
        image_hashes = {}
    weights_dir = args.output_dir / "weights"
    model, transform = clip.load("ViT-B/32", device=args.device, jit=False, download_root=str(weights_dir))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    weight_files = list(weights_dir.glob("*.pt"))
    if len(weight_files) != 1:
        raise ValueError("Expected one pinned CLIP ViT-B/32 checkpoint")
    weights_path = weight_files[0]
    identity = cache_identity(args.root, weights_path, args.box_policy)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_path = args.output_dir / "features.h5"
    partial_path = args.output_dir / "features.h5.partial"
    if final_path.exists() or (args.output_dir / "manifest.json").exists():
        raise FileExistsError("Completed visual cache already exists; use a new output directory")
    if not partial_path.exists():
        initialize_cache(partial_path, args.root, identity)
    loaders = {
        split: VG150AnnotationLoader(args.root, split, args.root / "splits.json", args.box_policy)
        for split in ("train", "val", "test")
    }
    completed = 0
    with h5py.File(partial_path, "r+") as cache:
        if json.loads(cache.attrs["identity"]) != identity:
            raise ValueError("Partial visual cache has different inputs or encoder weights")
        for split, loader in loaders.items():
            for index, row in enumerate(loader.indices):
                if bool(cache["done"][row]):
                    continue
                record = loader.load(index)
                image_path = args.root / "images" / f"{record.image_id}.jpg"
                if not image_path.is_file():
                    if args.existing_only:
                        continue
                    raise FileNotFoundError(image_path)
                if not args.existing_only and sha256(image_path) != image_hashes[row]:
                    raise ValueError(f"JPEG hash mismatch for image row {row}")
                with Image.open(image_path) as image:
                    if image.format != "JPEG":
                        raise ValueError(f"Not a JPEG: {image_path}")
                    image = image.convert("RGB")
                    boxes = [obj.bbox_xyxy for obj in record.objects]
                    scaled_boxes = scale_boxes_to_jpeg(boxes, (record.width, record.height), image.size)
                    vectors = extract_vectors(model, transform, image, scaled_boxes, args.batch_size, args.device)
                first, last = int(cache["row_first"][row]), int(cache["row_last"][row])
                if len(boxes) != last - first + 1:
                    raise ValueError(f"HDF5 object range mismatch for image row {row}")
                cache["features"][first : last + 1] = vectors.astype(np.float16)
                cache["boxes_xyxy"][first : last + 1] = np.asarray(boxes, dtype=np.float32)
                cache["done"][row] = True
                completed += 1
                if completed % 100 == 0:
                    cache.flush()
                    print(f"[CACHE] new={completed} split={split} row={row}", flush=True)
                if args.max_images is not None and completed >= args.max_images:
                    break
            if args.max_images is not None and completed >= args.max_images:
                break
        cache.flush()
        missing = [row for row in selected_rows if not bool(cache["done"][row])]
    if missing:
        print(f"[PARTIAL] new={completed}; remaining={len(missing)}; resume with the same command", flush=True)
        return
    partial_path.replace(final_path)
    manifest = {**identity, "complete": True, "images": len(selected_rows), "features_sha256": sha256(final_path)}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] {len(selected_rows)} images; cache={final_path}", flush=True)


if __name__ == "__main__":
    main()
