"""Repeatable real-crop CPU smoke check; never initializes CUDA."""

import argparse
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

from ..contracts import read_json
from .features import CONFIG_PATH, ClipEncoder, preprocess_crop
from .geometry import geometry_from_mask
from .media import MEDIA_ROOT, iter_video_frames, read_mask

CASES = [
    ("0001_4164158586", 1),
    ("P01_03", 2),
    ("03f2ed96-1719-427d-acf4-8bf504f1d66d", 1),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = read_json(CONFIG_PATH)
    index = read_json(MEDIA_ROOT / "media_index.json")
    crops, cases = [], []
    for video_id, track_id in CASES:
        media = index["videos"][video_id]
        decoder = iter_video_frames(MEDIA_ROOT / media["video_path"])
        try:
            rgb = next(decoder).to_ndarray(format="rgb24")
        finally:
            decoder.close()
        with zipfile.ZipFile(MEDIA_ROOT / media["mask_archive"]) as archive:
            mask = read_mask(archive, media["mask_members"][0])
        boxes, _, visible, _ = geometry_from_mask(mask, [track_id], allow_unannotated=True)
        if not visible[0]:
            raise ValueError(f"Probe track is absent: {video_id}/{track_id}")
        crops.append(preprocess_crop(rgb, mask, track_id, boxes[0], config))
        cases.append(dict(video_id=video_id, frame_id=0, track_id=track_id, source=media["source"]))
    started = time.monotonic()
    encoder = ClipEncoder(Path("data/external/pvsg_clip_v1/ViT-B-32.pt"), "cpu", config)
    loaded = time.monotonic()
    first = encoder(crops)
    finished = time.monotonic()
    second = encoder(crops)
    np.testing.assert_allclose(first, second, atol=1e-7, rtol=0)
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1, atol=1e-6, rtol=0)
    if (
        torch.cuda.is_initialized()
        or encoder.model.training
        or any(p.requires_grad for p in encoder.model.parameters())
    ):
        raise ValueError("CPU/frozen encoder contract violated")
    result = dict(
        device="cpu",
        model=config["model"],
        batch_size=len(crops),
        seed=config["seed"],
        cases=cases,
        feature_shape=list(first.shape),
        unit_norm_max_error=float(abs(np.linalg.norm(first, axis=1) - 1).max()),
        repeat_max_difference=float(abs(first - second).max()),
        load_seconds=loaded - started,
        forward_seconds=finished - loaded,
        gpu_initialized=False,
        training=False,
        torch=torch.__version__,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
