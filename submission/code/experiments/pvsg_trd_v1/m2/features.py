"""Frozen CLIP masked-crop cache; GPU execution is handed off to the user."""

import argparse
import contextlib
import fcntl
import json
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from ..contracts import read_json
from ..validate_m0 import canonical_sha256
from .download import sha256
from .media import MEDIA_ROOT, iter_video_timeline, read_mask

CONFIG_PATH = Path(__file__).parent / "configs/clip_masked_v1.json"


def preprocess_crop(rgb, mask, track_id, box, config):
    x0, y0, x1, y1 = map(int, box)
    if not 0 <= x0 < x1 <= rgb.shape[1] or not 0 <= y0 < y1 <= rgb.shape[0] or mask.shape != rgb.shape[:2]:
        raise ValueError("Crop/mask geometry mismatch")
    crop = rgb[y0:y1, x0:x1].copy()
    crop[mask[y0:y1, x0:x1] != track_id] = config["background_rgb"]
    image = Image.fromarray(crop)
    size = config["image_size"]
    width, height = image.size
    resized = (size, int(size * height / width)) if width <= height else (int(size * width / height), size)
    image = image.resize(resized, Image.Resampling.BICUBIC)
    left, top = int(round((image.width - size) / 2)), int(round((image.height - size) / 2))
    pixels = np.asarray(image.crop((left, top, left + size, top + size)), dtype=np.float32) / 255
    pixels = (pixels - np.array(config["mean"], dtype=np.float32)) / np.array(config["std"], dtype=np.float32)
    return np.ascontiguousarray(pixels.transpose(2, 0, 1))


class ClipEncoder:
    def __init__(self, checkpoint, device, config):
        import torch

        from .vendor.openai_clip_model import VisionTransformer

        if sha256(checkpoint) != config["checkpoint_sha256"]:
            raise ValueError("CLIP checkpoint SHA256 mismatch")
        if torch.__version__ != config["torch_version"]:
            raise ValueError(f"Expected torch {config['torch_version']}; found {torch.__version__}")
        torch.manual_seed(config["seed"])
        torch.set_num_threads(4)
        self.torch = torch
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Only CPU or CUDA execution is defined")
        pretrained = torch.jit.load(str(checkpoint), map_location="cpu").state_dict()
        visual = {
            name.removeprefix("visual."): tensor for name, tensor in pretrained.items() if name.startswith("visual.")
        }
        self.model = VisionTransformer(224, 32, 768, 12, 12, 512).float()
        self.model.load_state_dict(visual, strict=True)
        self.model.requires_grad_(False).eval().to(self.device)
        del pretrained, visual

    def __call__(self, crops):
        torch = self.torch
        images = torch.from_numpy(np.stack(crops)).to(self.device)
        precision = (
            torch.autocast("cuda", dtype=torch.float16) if self.device.type == "cuda" else contextlib.nullcontext()
        )
        with torch.inference_mode(), precision:
            features = self.model(images).float()
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        result = features.cpu().numpy()
        if result.shape != (len(crops), 512) or not np.isfinite(result).all():
            raise ValueError("Invalid encoder output")
        return result


def cache_identity(config, device, batch_size):
    from importlib.metadata import version

    import torch

    code = {
        str(p.relative_to(Path(__file__).parent)): sha256(p)
        for p in sorted(Path(__file__).parent.rglob("*.py"))
        if not p.name.startswith("test_")
    }
    return {
        "config_sha256": canonical_sha256(config),
        "code_sha256": code,
        "device_type": device.split(":")[0],
        "compute_precision": "fp16_autocast" if device.startswith("cuda") else "fp32",
        "batch_size": batch_size,
        "seed": config["seed"],
        "torch": torch.__version__,
        "numpy": version("numpy"),
        "Pillow": version("Pillow"),
        "av": version("av"),
    }


def extract_video(video_id, encoder, *, media, media_root, cache_root, output, config, identity, batch_size):
    record = read_json(cache_root / "records" / f"{video_id}.json")
    geometry_path = cache_root / "geometry" / f"{video_id}.npz"
    if (
        sha256(geometry_path) != record["geometry_sha256"]
        or sha256(media_root / media["video_path"]) != media["video_sha256"]
    ):
        raise ValueError(f"Input artifact changed: {video_id}")
    fingerprint = canonical_sha256({"identity": identity, "geometry_sha256": record["geometry_sha256"], "media": media})
    output.mkdir(parents=True, exist_ok=True)
    metadata_path = output / f"{video_id}.json"
    feature_path = output / f"{video_id}.npz"
    if metadata_path.exists():
        previous = read_json(metadata_path)
        if (
            previous["fingerprint"] != fingerprint
            or not feature_path.exists()
            or sha256(feature_path) != previous["feature_sha256"]
        ):
            raise ValueError(
                f"Existing feature cache has incompatible/corrupt provenance: {video_id}; use a new output version"
            )
        return previous
    started = time.monotonic()
    with np.load(geometry_path, allow_pickle=False) as data:
        visible = data["visible"]
        boxes = data["boxes"]
        ids = data["track_ids"]
        rgb_available = data["rgb_available"]
    positions = np.argwhere(visible.T)  # frame-major, then sorted track slot
    rows = len(positions)
    features = np.zeros((rows, 512), dtype=np.float16)
    available = rgb_available[positions[:, 0]] if rows else np.empty(0, dtype=bool)
    pending, pending_rows = [], []

    def flush():
        if pending:
            encoded = np.asarray(encoder(pending), dtype=np.float32)
            if encoded.shape != (len(pending), 512) or not np.isfinite(encoded).all():
                raise ValueError("Encoder output shape/finite check failed")
            norms = np.linalg.norm(encoded, axis=1)
            if not np.allclose(norms, 1, atol=1e-4):
                raise ValueError("Encoder output must be L2 normalized")
            features[pending_rows] = encoded.astype(np.float16)
            pending.clear()
            pending_rows.clear()

    cursor = 0
    decoder = iter_video_timeline(media_root / media["video_path"])
    with zipfile.ZipFile(media_root / media["mask_archive"]) as bundle:
        for frame_id in range(visible.shape[1]):
            frame = next(decoder, None)
            if (frame is not None) != bool(rgb_available[frame_id]):
                raise ValueError("RGB availability differs from audited media")
            first = cursor
            while cursor < rows and positions[cursor, 0] == frame_id:
                cursor += 1
            if frame is None or first == cursor:
                continue
            rgb = frame.to_ndarray(format="rgb24")
            mask = read_mask(bundle, media["mask_members"][frame_id])
            for row in range(first, cursor):
                slot = positions[row, 1]
                pending.append(preprocess_crop(rgb, mask, int(ids[slot]), boxes[slot, frame_id], config))
                pending_rows.append(row)
                if len(pending) == batch_size:
                    flush()
        flush()
        extra = sum(frame is not None for frame in decoder)
    if extra != record["extra_unannotated_rgb_frames"]:
        raise ValueError("RGB tail differs from audit")
    temporary = feature_path.with_suffix(".partial.npz")
    np.savez(
        temporary,
        features=features,
        frame_ids=positions[:, 0],
        track_ids=ids[positions[:, 1]],
        visual_available=available,
    )
    temporary.replace(feature_path)
    result = {
        "video_id": video_id,
        "fingerprint": fingerprint,
        "feature_sha256": sha256(feature_path),
        "rows": rows,
        "visual_rows": int(available.sum()),
        "missing_rgb_rows": int((~available).sum()),
        "dimension": 512,
        "dtype": "float16",
        "geometry_sha256": record["geometry_sha256"],
        "wall_seconds": time.monotonic() - started,
        "bytes": feature_path.stat().st_size,
        "identity": identity,
    }
    temporary = metadata_path.with_suffix(".partial.json")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(metadata_path)
    print(f"Cached {video_id}: {rows} rows ({result['visual_rows']} visual)", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--checkpoint", type=Path, default=Path("data/external/pvsg_clip_v1/ViT-B-32.pt"))
    parser.add_argument("--media-root", type=Path, default=MEDIA_ROOT)
    parser.add_argument("--cache-root", type=Path, default=Path("data/external/pvsg_m2_cache_v4"))
    parser.add_argument("--output", type=Path, default=Path("data/external/pvsg_clip_cache_v1"))
    parser.add_argument(
        "--video-id", action="append", help="Optional explicit CPU/pilot subset; report is marked partial"
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("batch-size must be positive")
    config = read_json(CONFIG_PATH)
    audit = read_json(args.cache_root / "media_audit.json")
    if audit["passed"] != 400 or audit["errors"]:
        raise ValueError("Complete media audit is required")
    index = read_json(args.media_root / "media_index.json")
    if sha256(args.media_root / "media_index.json") != audit["media_index_sha256"]:
        raise ValueError("Media index drift")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        encoder = ClipEncoder(args.checkpoint, args.device, config)
        identity = cache_identity(config, args.device, args.batch_size)
        ids = sorted(args.video_id or index["videos"])
        if len(ids) != len(set(ids)) or not set(ids) <= set(index["videos"]):
            raise ValueError("Invalid video selection")
        results = [
            extract_video(
                video_id,
                encoder,
                media=index["videos"][video_id],
                media_root=args.media_root,
                cache_root=args.cache_root,
                output=args.output,
                config=config,
                identity=identity,
                batch_size=args.batch_size,
            )
            for video_id in ids
        ]
        summary = {
            "feature_id": config["feature_id"],
            "identity": identity,
            "videos": len(results),
            "complete_cohort": set(ids) == set(index["videos"]),
            "rows": sum(r["rows"] for r in results),
            "visual_rows": sum(r["visual_rows"] for r in results),
            "bytes": sum(r["bytes"] for r in results),
            "training": False,
            "automatic_next_run": False,
            "video_ids": ids,
            "feature_hashes": {r["video_id"]: r["feature_sha256"] for r in results},
        }
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(
            json.dumps({k: v for k, v in summary.items() if k not in ("identity", "feature_hashes", "video_ids")}),
            flush=True,
        )


if __name__ == "__main__":
    main()
