"""Deterministic overlap merge and multi-label decoding under the M0 defaults."""

import hashlib
from dataclasses import dataclass

import numpy as np

from extension_pvsg.contracts import NUM_PREDICATES, PROTOCOL_ID, normalize_predictions, ordered_candidates, validate_video, visibility_masks
from extension_pvsg.intervals import mask_to_intervals


@dataclass(frozen=True)
class WindowLogits:
    pairs: np.ndarray  # [P,2], original ordered track IDs
    frame_indices: np.ndarray  # [L], original zero-based frame indices
    logits: np.ndarray  # [P,L,57], padding already removed


def merge_window_logits(video, windows):
    video = validate_video(video)
    pairs = ordered_candidates(video)
    pair_index = {pair: i for i, pair in enumerate(pairs)}
    total = video["num_frames"]
    checked = []
    for window in windows:
        local_pairs, frames = np.asarray(window.pairs), np.asarray(window.frame_indices)
        logits = np.asarray(window.logits, dtype=np.float64)
        if local_pairs.ndim != 2 or local_pairs.shape[1] != 2 or local_pairs.dtype.kind not in "iu":
            raise ValueError("Window pairs must be an integer [P,2] array")
        keys = [tuple(map(int, pair)) for pair in local_pairs]
        if len(set(keys)) != len(keys) or any(pair not in pair_index for pair in keys):
            raise ValueError("Window has duplicate/non-candidate track pairs")
        if frames.ndim != 1 or frames.dtype.kind not in "iu" or not len(frames):
            raise ValueError("Window frame indices must be a nonempty integer vector")
        if len(set(frames.tolist())) != len(frames) or frames.min() < 0 or frames.max() >= total:
            raise ValueError("Window has duplicate/out-of-range frame indices")
        if logits.shape != (len(keys), len(frames), NUM_PREDICATES) or not np.isfinite(logits).all():
            raise ValueError("Window logits must be finite [P,L,57]")
        # Canonical accumulation order prevents arrival order from changing a
        # threshold decision due to floating-point addition round-off.
        order_key = (tuple(frames.tolist()), tuple(keys), hashlib.sha256(logits.tobytes()).hexdigest())
        checked.append((order_key, keys, frames, logits))
    sums = np.zeros((len(pairs), total, NUM_PREDICATES), dtype=np.float64)
    counts = np.zeros((len(pairs), total), dtype=np.int64)
    for _, keys, frames, logits in sorted(checked, key=lambda item: item[0]):
        for i, pair in enumerate(keys):
            index = pair_index[pair]
            sums[index, frames] += logits[i]
            counts[index, frames] += 1
    visibility = visibility_masks(video)
    valid = np.asarray([visibility[s] & visibility[o] for s, o in pairs], dtype=bool).reshape(len(pairs), total)
    if np.any(valid & (counts == 0)):
        raise ValueError("Window predictions do not cover every covisible candidate/frame")
    merged = np.divide(sums, counts[..., None], out=np.zeros_like(sums), where=counts[..., None] > 0)
    if not np.isfinite(merged).all():
        raise ValueError("Non-finite overlap accumulation")
    return {
        "pairs": np.asarray(pairs, dtype=np.int64).reshape(-1, 2),
        "logits": merged,
        "counts": counts,
        "valid": valid,
    }


def decode_windows(video, windows):
    """Mean logits -> sigmoid >= .5 -> inclusive intervals; no gap merging.

    No target argument. All candidates must be covered, including negative pairs.
    Alternative thresholds/decoders require a separately versioned run config.
    """
    video = validate_video(video)
    merged = merge_window_logits(video, windows)
    z = merged["logits"]
    exp = np.exp(-np.abs(z))
    probabilities = np.where(z >= 0, 1 / (1 + exp), exp / (1 + exp))
    active = (probabilities >= 0.5) & merged["valid"][..., None]
    rows = []
    for i, (s, o) in enumerate(merged["pairs"]):
        for predicate in range(NUM_PREDICATES):
            support = active[i, :, predicate]
            if support.any():
                rows.append(
                    {
                        "subject_id": int(s),
                        "object_id": int(o),
                        "predicate_id": predicate,
                        "intervals": mask_to_intervals(support),
                        "score": float(probabilities[i, support, predicate].mean()),
                    }
                )
    return normalize_predictions(
        {
            "schema_version": 1,
            "kind": "relation_predictions",
            "protocol_id": PROTOCOL_ID,
            "video_id": video["video_id"],
            "num_frames": video["num_frames"],
            "predictions": rows,
        },
        video,
    )
