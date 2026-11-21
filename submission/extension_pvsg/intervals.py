"""Zero-based inclusive frame intervals. No implicit clipping or frame offsets."""

import numpy as np


def integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}: {value!r}")
    return int(value)


def normalize_intervals(intervals, num_frames):
    """Validate then union overlapping/adjacent intervals; empty support is allowed."""
    integer(num_frames, "num_frames", 1)
    checked = []
    for span in intervals:
        if len(span) != 2:
            raise ValueError("An interval must contain exactly start and end")
        start, end = (integer(v, "interval endpoint") for v in span)
        if not start <= end < num_frames:
            raise ValueError(f"Invalid inclusive interval [{start}, {end}] for T={num_frames}")
        checked.append((start, end))
    merged = []
    for start, end in sorted(checked):
        if merged and start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def intervals_to_mask(intervals, num_frames):
    mask = np.zeros(integer(num_frames, "num_frames", 1), dtype=bool)
    for start, end in normalize_intervals(intervals, num_frames):
        mask[start : end + 1] = True
    return mask


def mask_to_intervals(mask):
    mask = np.asarray(mask)
    if mask.ndim != 1 or mask.dtype != np.bool_:
        raise ValueError("Timeline must be a one-dimensional boolean array")
    transitions = np.diff(np.concatenate(([False], mask, [False])).astype(np.int8))
    return [
        [int(a), int(b - 1)]
        for a, b in zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1), strict=False)
    ]


def temporal_iou(first, second, num_frames):
    """True temporal set IoU, including prediction frames outside GT support."""
    first = intervals_to_mask(first, num_frames)
    second = intervals_to_mask(second, num_frames)
    union = np.count_nonzero(first | second)
    return float(np.count_nonzero(first & second) / union) if union else 0.0


def window_bounds(num_frames, size, stride):
    """Cover the full timeline, retaining a final partial window if needed."""
    integer(num_frames, "num_frames", 1)
    integer(size, "size", 1)
    integer(stride, "stride", 1)
    if stride > size:
        raise ValueError("stride > size leaves uncovered frames")
    bounds = []
    for start in range(0, num_frames, stride):
        stop = min(start + size, num_frames)
        bounds.append((start, stop))
        if stop == num_frames:
            break
    return bounds
