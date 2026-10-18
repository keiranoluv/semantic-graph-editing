"""Compute graph statistics from VG150 HDF5 annotations."""

import argparse
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np


def count_pairs(pairs: np.ndarray, num_objects: int) -> tuple[int, int, int]:
    """Return positive, ambiguous, and background ordered-pair counts."""
    candidates = num_objects * (num_objects - 1)
    labels_by_pair = {}

    for subject, obj, predicate in pairs:
        subject, obj, predicate = int(subject), int(obj), int(predicate)

        if not (0 <= subject < num_objects and 0 <= obj < num_objects):
            raise ValueError("Relationship endpoint outside object range")
        if predicate <= 0:
            raise ValueError("Predicate ID must be positive")

        if subject == obj:
            continue

        labels_by_pair.setdefault((subject, obj), set()).add(predicate)

    positive = len(labels_by_pair)
    ambiguous = sum(len(labels) > 1 for labels in labels_by_pair.values())
    background = candidates - positive

    return positive, ambiguous, background


def summarize_split(h5: h5py.File, indices: list[int]) -> dict:
    """Summarize selected image rows without loading image pixels."""
    first_box = h5["img_to_first_box"][:].reshape(-1)
    last_box = h5["img_to_last_box"][:].reshape(-1)
    first_rel = h5["img_to_first_rel"][:].reshape(-1)
    last_rel = h5["img_to_last_rel"][:].reshape(-1)

    object_labels = h5["labels"][:].reshape(-1)
    predicates = h5["predicates"][:].reshape(-1)
    relationships = h5["relationships"][:]

    object_hist = Counter()
    predicate_hist = Counter()

    totals = Counter()
    max_objects = 0

    for row in indices:
        start = int(first_box[row])
        end = int(last_box[row])

        if start < 0 or end < start:
            raise ValueError(f"Invalid object range at HDF5 row {row}")

        num_objects = end - start + 1
        max_objects = max(max_objects, num_objects)
        totals["objects"] += num_objects
        totals["images"] += 1

        object_hist.update(int(v) for v in object_labels[start : end + 1])

        rel_start = int(first_rel[row])
        rel_end = int(last_rel[row])

        if rel_start < 0:
            if rel_end != -1:
                raise ValueError(f"Invalid relation range at row {row}")
            local_relations = np.empty((0, 3), dtype=np.int64)
        else:
            if rel_end < rel_start:
                raise ValueError(f"Invalid relation range at row {row}")

            image_pairs = relationships[rel_start : rel_end + 1]
            image_predicates = predicates[rel_start : rel_end + 1]

            local_pairs = image_pairs.astype(np.int64) - start

            local_relations = np.column_stack((local_pairs, image_predicates))
            predicate_hist.update(int(v) for v in image_predicates)

        num_relations = len(local_relations)
        candidates = num_objects * (num_objects - 1)
        positive, ambiguous, background = count_pairs(local_relations, num_objects)

        totals["relationships"] += num_relations
        totals["candidate_pairs"] += candidates
        totals["positive_pairs"] += positive
        totals["ambiguous_pairs"] += ambiguous
        totals["background_pairs"] += background

    images = totals["images"]
    candidate_pairs = totals["candidate_pairs"]

    return {
        "num_images": images,
        "num_objects": totals["objects"],
        "num_relationships": totals["relationships"],
        "num_candidate_pairs": candidate_pairs,
        "num_positive_pairs": totals["positive_pairs"],
        "num_ambiguous_pairs": totals["ambiguous_pairs"],
        "num_background_pairs": totals["background_pairs"],
        "background_ratio": (totals["background_pairs"] / candidate_pairs if candidate_pairs else 0.0),
        "avg_objects_per_image": (totals["objects"] / images if images else 0.0),
        "avg_relationships_per_image": (totals["relationships"] / images if images else 0.0),
        "max_objects_per_image": max_objects,
        "object_class_counts": dict(sorted(object_hist.items())),
        "predicate_class_counts": dict(sorted(predicate_hist.items())),
    }


def main() -> None:
    """Compute statistics for all three manifest splits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/vg150"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/vg150_graph_statistics.json"),
    )
    args = parser.parse_args()

    manifest = json.loads((args.root / "splits.json").read_text())

    report = {
        "dataset": "VG150",
        "candidate_policy": "all_ordered_pairs_except_self",
        "splits": {},
    }

    with h5py.File(args.root / "VG-SGG-with-attri.h5", "r") as h5:
        for split in ("train", "val", "test"):
            print(f"Processing {split}...", flush=True)
            stats = summarize_split(h5, manifest[split])
            report["splits"][split] = stats

            print(
                f"[{split}] images={stats['num_images']}, "
                f"objects={stats['num_objects']}, "
                f"relations={stats['num_relationships']}, "
                f"background={stats['background_ratio']:.2%}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[OK] Report saved to {args.output}")


if __name__ == "__main__":
    main()
