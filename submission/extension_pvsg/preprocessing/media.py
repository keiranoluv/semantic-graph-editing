"""Read pinned media archives without extracting all RGB frames or masks."""

import io
from pathlib import Path

import av
import numpy as np
from PIL import Image

MEDIA_ROOT = Path("data/external/pvsg_media_v1")


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
