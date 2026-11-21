"""Read pinned media archives without extracting all RGB frames or masks."""

import argparse
import hashlib
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

import av
import numpy as np
from PIL import Image

from extension_pvsg.contracts import ROOT, read_json
from extension_pvsg.m2.download import sha256

MEDIA_ROOT = Path("data/external/pvsg_media_v1")
MEDIA_LOCK = Path(__file__).parent / "manifests/media.lock.json"


def archive_members(archive, kind):
    """Index only basename video IDs or video/frame masks; reject collisions."""
    by_video = defaultdict(dict)
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            path = PurePosixPath(info.filename)
            if info.is_dir() or "__MACOSX" in path.parts or path.name.startswith("."):
                continue
            if ".." in path.parts or path.is_absolute():
                raise ValueError("Unsafe archive path")
            if kind == "masks" and path.suffix.lower() == ".png":
                video_id, key = path.parent.name, int(path.stem)
            elif kind == "videos" and path.suffix.lower() == ".mp4":
                video_id, key = path.stem, "video"
            else:
                continue
            if key in by_video[video_id]:
                raise ValueError(f"Archive member collision: {video_id}/{key}")
            by_video[video_id][key] = {"member": info.filename, "bytes": info.file_size, "crc32": info.CRC}
    return dict(by_video)


def index_media(root=MEDIA_ROOT):
    root = Path(root)
    lock = read_json(MEDIA_LOCK)
    split = read_json(ROOT / "manifests/pvsg_author_release_338_62.lock.json")
    indices = {}
    for item in lock["archives"]:
        archive = root / "archives" / item["filename"]
        if archive.stat().st_size != item["bytes"] or sha256(archive) != item["sha256"]:
            raise ValueError(f"Media archive checksum mismatch: {archive}")
        indices[item["source"], item["kind"]] = (item, archive_members(archive, item["kind"]))
    records = {}
    for video_id, source in sorted(split["video_sources"].items()):
        mask_item, masks = indices[source, "masks"]
        video_item, videos = indices[source, "videos"]
        frames = sorted(masks[video_id])
        if frames != list(range(len(frames))) or not frames:
            raise ValueError(f"Non-contiguous zero-based mask frames: {video_id}")
        member = videos[video_id]["video"]
        destination = root / "videos" / source / f"{video_id}.mp4"
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        # Always verify the extracted content against the archive member; this
        # also checks its ZIP CRC. Never call extractall on upstream paths.
        with zipfile.ZipFile(root / "archives" / video_item["filename"]) as archive:
            with archive.open(member["member"]) as stream:
                if destination.exists():
                    while chunk := stream.read(8 * 1024 * 1024):
                        digest.update(chunk)
                    if destination.stat().st_size != member["bytes"] or sha256(destination) != digest.hexdigest():
                        raise ValueError(f"Extracted video differs from archive: {video_id}")
                else:
                    temporary = destination.with_suffix(".mp4.partial")
                    with temporary.open("wb") as output:
                        while chunk := stream.read(8 * 1024 * 1024):
                            output.write(chunk)
                            digest.update(chunk)
                    temporary.replace(destination)
        with av.open(str(destination)) as container:
            stream = container.streams.video[0]
            header = {
                "frames": stream.frames,
                "width": stream.width,
                "height": stream.height,
                "fps": float(stream.average_rate) if stream.average_rate else None,
            }
        records[video_id] = {
            "source": source,
            "video_path": str(destination.relative_to(root)),
            "video_sha256": digest.hexdigest(),
            "video_bytes": member["bytes"],
            "mask_archive": f"archives/{mask_item['filename']}",
            "mask_members": [masks[video_id][frame]["member"] for frame in frames],
            "mask_count": len(frames),
            "header": header,
        }
    extras = {
        f"{source}/{kind}": sorted(set(entries) - {vid for vid, src in split["video_sources"].items() if src == source})
        for (source, kind), (_, entries) in indices.items()
    }
    if any(extras.values()):
        raise ValueError(f"Unexpected media video IDs: {extras}")
    result = {"schema_version": 1, "media_lock_sha256": sha256(MEDIA_LOCK), "videos": records}
    (root / "media_index.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def read_mask(bundle, member):
    with Image.open(io.BytesIO(bundle.read(member))) as image:
        if image.mode not in ("P", "L", "I", "I;16"):
            raise ValueError(f"Unsupported ID mask mode: {image.mode}")
        return np.array(image)


def iter_video_frames(path):
    """CPU decode in presentation order; no sampling or hidden resize."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        for frame in container.decode(stream):
            yield frame


def iter_video_timeline(path):
    """Align presentation timestamps to the 5 FPS mask grid; gaps stay absent."""
    origin = None
    next_index = 0
    for frame in iter_video_frames(path):
        if frame.time is None:
            raise ValueError("Missing frame presentation timestamp")
        seconds = float(frame.time)
        if origin is None:
            origin = seconds
        index = round((seconds - origin) * 5)
        if index < next_index or abs(seconds - origin - index / 5) > 0.021:
            raise ValueError("Nonmonotonic or off-grid 5 FPS frame timestamp")
        while next_index < index:
            yield None
            next_index += 1
        yield frame
        next_index += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=MEDIA_ROOT)
    args = parser.parse_args()
    result = index_media(args.root)
    print(
        json.dumps(
            {
                "indexed_videos": len(result["videos"]),
                "mask_frames": sum(x["mask_count"] for x in result["videos"].values()),
            }
        )
    )
