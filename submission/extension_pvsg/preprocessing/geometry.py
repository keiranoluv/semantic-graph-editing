"""Mask-derived geometry only; no predicates or learned statistics."""

import numpy as np
from scipy.ndimage import find_objects


def geometry_from_mask(mask, track_ids, *, allow_unannotated=False):
    mask = np.asarray(mask)
    ids = np.asarray(track_ids, dtype=np.int64)
    if mask.ndim != 2 or mask.dtype.kind not in "iu" or not len(ids) or np.any(ids <= 0):
        raise ValueError("M2 masks require 2D integer IDs, explicit void=0 and positive annotated track IDs")
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("Duplicate geometry track IDs")
    counts = np.bincount(mask.ravel().astype(np.int64), minlength=int(max(ids.max(), mask.max())) + 1)
    unknown = set(np.flatnonzero(counts).tolist()) - set(ids.tolist()) - {0}
    if unknown and not allow_unannotated:
        raise ValueError(f"Unannotated mask IDs: {sorted(unknown)}")
    slices = find_objects(mask, max_label=int(ids.max()))
    height, width = mask.shape
    boxes = np.zeros((len(ids), 4), dtype=np.int32)
    geometry = np.zeros((len(ids), 6), dtype=np.float32)
    visible = counts[ids] > 0
    for i, track_id in enumerate(ids):
        if not visible[i]:
            continue
        ys, xs = slices[track_id - 1]
        x0, x1, y0, y1 = xs.start, xs.stop, ys.start, ys.stop
        boxes[i] = [x0, y0, x1, y1]  # pixel edges, right/bottom exclusive
        w, h = (x1 - x0) / width, (y1 - y0) / height
        geometry[i] = [
            (x0 + x1) / (2 * width),
            (y0 + y1) / (2 * height),
            w,
            h,
            w * h,
            counts[track_id] / (width * height),
        ]
    return boxes, geometry, visible, {int(key): int(counts[key]) for key in sorted(unknown)}


def motion_from_geometry(geometry, visible):
    """Velocity per annotation frame between last/current valid observations."""
    motion = np.zeros((*visible.shape, 4), dtype=np.float32)
    delta_frames = np.zeros(visible.shape, dtype=np.int32)
    for track in range(len(visible)):
        frames = np.flatnonzero(visible[track])
        if len(frames) > 1:
            previous, current = frames[:-1], frames[1:]
            delta = current - previous
            motion[track, current] = (geometry[track, current, :4] - geometry[track, previous, :4]) / delta[:, None]
            delta_frames[track, current] = delta
    return motion, delta_frames
