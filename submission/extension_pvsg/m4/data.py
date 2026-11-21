"""Add target-free contiguous pair sequences to the frozen M3 batch contract."""

import numpy as np
import torch

from extension_pvsg.m3.data import collate as collate_m3


def collate(samples):
    batch = collate_m3(samples)
    frames = batch["query_frame_indices"].numpy()
    pairs = batch["query_track_pairs"].numpy()
    windows = batch["query_batch"].numpy()
    count = len(frames)
    # A window ID is essential: even overlapping windows from the same video
    # must remain independent. Track IDs are grouping keys, never features.
    order = np.lexsort((frames, pairs[:, 1], pairs[:, 0], windows))
    groups = np.column_stack((windows, pairs))[order]
    cuts = np.flatnonzero(np.any(groups[1:] != groups[:-1], axis=1) | (np.diff(frames[order]) != 1)) + 1
    buckets = {}
    if count:
        for segment in np.split(order, cuts):
            width = 1 << (len(segment) - 1).bit_length()
            buckets.setdefault(width, []).append(segment)
    flattened = []
    for width, segments in sorted(buckets.items()):
        indices = np.full((len(segments), width), count, dtype=np.int64)
        for row, segment in enumerate(segments):
            indices[row, : len(segment)] = segment
            flattened.extend(segment)
        batch[f"sequence_indices_{width}"] = torch.from_numpy(indices)
    batch["sequence_widths"] = tuple(sorted(buckets))
    batch["sequence_inverse"] = torch.from_numpy(np.argsort(np.asarray(flattened, dtype=np.int64)))
    return batch
