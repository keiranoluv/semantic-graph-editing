"""Small artificial fixture; labels/logits here are never model benchmarks."""

import numpy as np

from .data import VOCABULARY, parse_targets, parse_video, rasterize_targets, visibility_from_masks
from .decoding import WindowLogits
from .intervals import window_bounds


def synthetic_record():
    return {
        "video_id": "synthetic_m1",
        "meta": {"num_frames": 10, "fps": 5, "height": 2, "width": 3},
        "objects": [
            {"object_id": 7, "category": VOCABULARY["object_classes"][0], "is_thing": True},
            {"object_id": 12, "category": VOCABULARY["object_classes"][115], "is_thing": False},
            {"object_id": 30, "category": VOCABULARY["object_classes"][1], "is_thing": True},
        ],
        "relations": [
            [7, 12, VOCABULARY["predicates"][0], [[2, 3], [7, 8]]],
            [7, 12, VOCABULARY["predicates"][1], [[3, 4]]],
        ],
        "captions": ["This field must never enter graph inputs."],
        "qa_pairs": [],
        "summary": "not a feature",
    }


def synthetic_masks():
    masks = np.zeros((10, 2, 3), dtype=np.int64)
    masks[:, 0, 0] = 7
    masks[:, 0, 1] = 12
    masks[5:7, 0, 1] = 0
    masks[:, 1, 0] = 30
    return masks


def synthetic_video_targets():
    record = synthetic_record()
    visibility = visibility_from_masks(record, synthetic_masks(), ignored_mask_ids=(0,))
    video = parse_video(record, source="vidor", visibility=visibility, mask_reference="synthetic://m1-masks")
    return video, parse_targets(record, video)


def perfect_fixture_windows(video, targets):
    """Deliberately use synthetic GT to test decoder plumbing, not inference."""
    windows = []
    for start, stop in window_bounds(video["num_frames"], size=6, stride=4):
        batch = rasterize_targets(video, targets, start=start, stop=stop)
        windows.append(WindowLogits(batch["pairs"], batch["frame_indices"], np.where(batch["labels"] > 0, 20.0, -20.0)))
    return windows
