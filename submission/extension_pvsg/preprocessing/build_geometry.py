"""CPU audit: decode every frame/mask, cache geometry, save 12-video overlays."""

import argparse
import hashlib
import json
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from common.io import sha256
from extension_pvsg.contracts import ROOT, read_json
from extension_pvsg.data import CLASS_IDS, VOCABULARY, parse_video
from extension_pvsg.intervals import mask_to_intervals
from extension_pvsg.preprocessing.geometry import geometry_from_mask, motion_from_geometry
from extension_pvsg.preprocessing.media import MEDIA_ROOT, iter_video_timeline, read_mask
from extension_pvsg.validation import canonical_sha256, validate_annotation_identity
from PIL import Image, ImageDraw


def overlay(rgb, mask, objects, title):
    rgb = np.asarray(rgb)
    colored = rgb.astype(np.float32).copy()
    for obj in objects:
        track_id = obj["object_id"]
        color = np.frombuffer(hashlib.sha256(str(track_id).encode()).digest()[:3], dtype=np.uint8)
        support = mask == track_id
        colored[support] = 0.65 * colored[support] + 0.35 * color
    image = Image.fromarray(colored.astype(np.uint8))
    image.thumbnail((420, 250))
    canvas = Image.new("RGB", (420, 290), "white")
    canvas.paste(image, ((420 - image.width) // 2, 36))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 3), title, fill="black")
    draw.text((4, 17), "GT ID masks overlay; RGB aligned by 5 FPS timestamps", fill="black")
    return canvas


def selected_overlays(manifest):
    train = manifest["outer_split"]["train_ids"]
    return [
        vid
        for source in ("vidor", "epic_kitchen", "ego4d")
        for vid in sorted(
            [v for v in train if manifest["video_sources"][v] == source],
            key=lambda v: hashlib.sha256(f"17:{source}:{v}".encode()).hexdigest(),
        )[:4]
    ]


def audit_one(video_id, record, media, root, output, bundle, make_overlay, code_hash):
    started = time.monotonic()
    fingerprint = canonical_sha256(
        {"media": media, "meta": record["meta"], "objects": record["objects"], "code_hash": code_hash}
    )
    completion = output / "records" / f"{video_id}.json"
    cache = output / "geometry" / f"{video_id}.npz"
    if completion.exists():
        previous = read_json(completion)
        if previous["fingerprint"] != fingerprint:
            raise ValueError(f"Audit provenance changed; use a new output version: {video_id}")
        if previous.get("status") == "passed" and cache.exists() and sha256(cache) == previous["geometry_sha256"]:
            return previous
    meta = record["meta"]
    frames = meta["num_frames"]
    header = media["header"]
    if media["mask_count"] != frames:
        raise ValueError(f"Frame/mask/metadata count mismatch for {video_id}")
    if (header["width"], header["height"]) != (meta["width"], meta["height"]) or abs(header["fps"] - 5) > 1e-6:
        raise ValueError(f"Video header dimensions/FPS mismatch for {video_id}")
    objects = sorted(record["objects"], key=lambda obj: obj["object_id"])
    ids = np.array([obj["object_id"] for obj in objects], dtype=np.int64)
    boxes = np.zeros((len(ids), frames, 4), dtype=np.int32)
    geometry = np.zeros((len(ids), frames, 6), dtype=np.float32)
    visible = np.zeros((len(ids), frames), dtype=bool)
    rgb_available = np.zeros(frames, dtype=bool)
    unannotated_regions = {}
    snapshots = []
    selected_frames = {0, frames // 2, frames - 1}
    count = 0
    decoder = iter_video_timeline(root / media["video_path"])
    for t in range(frames):
        frame = next(decoder, None)
        if frame is not None:
            if frame.width != meta["width"] or frame.height != meta["height"]:
                raise ValueError(f"Decoded dimensions mismatch: {video_id}/{t}")
            rgb_available[t] = True
        mask = read_mask(bundle, media["mask_members"][t])
        if mask.shape != (meta["height"], meta["width"]):
            raise ValueError(f"Mask dimensions mismatch: {video_id}/{t}")
        boxes[:, t], geometry[:, t], visible[:, t], unknown = geometry_from_mask(mask, ids, allow_unannotated=True)
        for track_id, pixels in unknown.items():
            item = unannotated_regions.setdefault(str(track_id), {"frames": 0, "pixels": 0})
            item["frames"] += 1
            item["pixels"] += pixels
        if make_overlay and t in selected_frames:
            path = output / "overlays" / f"{video_id}_{t:04d}.jpg"
            rgb = (
                frame.to_ndarray(format="rgb24")
                if frame is not None
                else np.full((*mask.shape, 3), 128, dtype=np.uint8)
            )
            title = f"{video_id} / frame {t}" if frame is not None else f"NO RGB: {video_id} / {t}"
            overlay(rgb, mask, objects, title).save(path, quality=90)
            snapshots.append(str(path.relative_to(output)))
        count += 1
    if count != frames:
        raise ValueError(f"Decoded {count}/{frames} frames: {video_id}")
    extra_rgb_frames = sum(frame is not None for frame in decoder)
    visibility = {int(track_id): mask_to_intervals(visible[i]) for i, track_id in enumerate(ids)}
    video = parse_video(
        record,
        source=media["source"],
        visibility=visibility,
        mask_reference=f"pvsg-zip://{media['mask_archive']}#{video_id}",
    )
    motion, delta_frames = motion_from_geometry(geometry, visible)
    temporary = cache.with_suffix(".partial.npz")
    np.savez_compressed(
        temporary,
        track_ids=ids,
        class_ids=np.array([CLASS_IDS[obj["category"]] for obj in objects]),
        boxes=boxes,
        geometry=geometry,
        visible=visible,
        motion=motion,
        delta_frames=delta_frames,
        rgb_available=rgb_available,
    )
    temporary.replace(cache)
    report = {
        "video_id": video_id,
        "status": "passed",
        "fingerprint": fingerprint,
        "source": media["source"],
        "mask_frames": count,
        "decoded_rgb_frames": int(rgb_available.sum()) + extra_rgb_frames,
        "missing_rgb_frame_ids": np.flatnonzero(~rgb_available).tolist(),
        "extra_unannotated_rgb_frames": extra_rgb_frames,
        "unannotated_mask_regions_ignored": unannotated_regions,
        "visible_observations": int(visible.sum()),
        "max_objects_per_frame": int(visible.sum(axis=0).max()),
        "absent_track_ids": [int(ids[i]) for i in range(len(ids)) if not visible[i].any()],
        "geometry_sha256": sha256(cache),
        "input": video,
        "overlays": snapshots,
        "wall_seconds": time.monotonic() - started,
    }
    temporary = completion.with_suffix(".partial.json")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(completion)
    print(f"Audited {video_id}: {frames} frames, {int(visible.sum())} visible objects", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=MEDIA_ROOT)
    parser.add_argument("--annotation", type=Path, default=Path("data/external/pvsg_m0/pvsg.hf_7e5f1ec.json"))
    parser.add_argument("--output", type=Path, default=Path("data/external/pvsg_m2_cache_v4"))
    parser.add_argument("--workers", type=int, choices=(1, 2, 4), default=4)
    args = parser.parse_args()
    manifest = read_json(ROOT / "manifests/pvsg_author_release_338_62.lock.json")
    validate_annotation_identity(args.annotation, manifest, VOCABULARY)
    records = {record["video_id"]: record for record in read_json(args.annotation)["data"]}
    index = read_json(args.root / "media_index.json")
    if set(index["videos"]) != set(manifest["video_sources"]):
        raise ValueError("Media index must cover exactly all 400 videos")
    for subdir in ("records", "geometry", "overlays"):
        (args.output / subdir).mkdir(parents=True, exist_ok=True)
    selected = set(selected_overlays(manifest))
    selected.update(vid for vid, media in index["videos"].items() if media["header"]["frames"] != media["mask_count"])
    code_hash = canonical_sha256(
        {name: sha256(Path(__file__).parent / name) for name in ("build_geometry.py", "geometry.py", "media.py")}
    )
    started = time.monotonic()

    def run(video_id):
        media = index["videos"][video_id]
        try:
            # Independent handles avoid shared seek/CRC/decompression state
            # when several CPU workers read the same upstream ZIP archive.
            with zipfile.ZipFile(args.root / media["mask_archive"]) as bundle:
                return audit_one(
                    video_id,
                    records[video_id],
                    media,
                    args.root,
                    args.output,
                    bundle,
                    video_id in selected,
                    code_hash,
                )
        except Exception as error:
            print(f"FAILED {video_id}: {error}", flush=True)
            return {"video_id": video_id, "status": "failed", "error": f"{type(error).__name__}: {error}"}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run, sorted(records)))
    passed = [r for r in results if r["status"] == "passed"]
    summary = {
        "schema_version": 1,
        "annotation_sha256": manifest["annotation"]["sha256"],
        "media_index_sha256": sha256(args.root / "media_index.json"),
        "code_hash": code_hash,
        "videos": len(results),
        "passed": len(passed),
        "errors": [r for r in results if r["status"] != "passed"],
        "mask_frames": sum(r["mask_frames"] for r in passed),
        "decoded_rgb_frames": sum(r["decoded_rgb_frames"] for r in passed),
        "missing_rgb_frames": sum(len(r["missing_rgb_frame_ids"]) for r in passed),
        "extra_unannotated_rgb_frames": sum(r["extra_unannotated_rgb_frames"] for r in passed),
        "videos_with_unannotated_mask_regions": [
            r["video_id"] for r in passed if r["unannotated_mask_regions_ignored"]
        ],
        "visible_observations": sum(r["visible_observations"] for r in passed),
        "overlay_video_ids": sorted(selected),
        "wall_seconds": time.monotonic() - started,
    }
    (args.output / "media_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    if summary["errors"]:
        raise SystemExit(2)
    (args.output / "inputs.json").write_text(json.dumps([r["input"] for r in results], indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
