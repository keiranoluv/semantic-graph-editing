"""List deterministic TRAIN image examples for reviewing VG150 object senses."""

import argparse
import csv
import io
import json
import random
from pathlib import Path

import h5py
import numpy as np

from scripts.prepare_knowledge import _save_outputs
from src.datasets.vg150 import VG150AnnotationLoader, _clip_box_to_image
from src.knowledge.entity_linking import _read_classes
from src.knowledge.transe.core import sha256


def sample_examples(indices, first_box, last_box, labels, metadata, *, boxes_1024=None, seed=2026, per_class=5):
    """Read train object labels/boxes only; never inspect predicates."""
    if per_class <= 0:
        raise ValueError("per_class must be positive")
    sampled = {i: [] for i in range(1, 151)}
    order = list(indices)
    random.Random(seed).shuffle(order)
    for row in order:
        start, end = int(first_box[row]), int(last_box[row])
        if start < 0 or end < start:
            continue
        image = metadata[row]
        for class_id in set(int(value) for value in labels[start : end + 1]):
            if class_id in sampled and len(sampled[class_id]) < per_class:
                entry = {"image_id": int(image["image_id"]), "annotation_row": row, "url": image.get("url")}
                if boxes_1024 is not None:
                    local = int(np.flatnonzero(labels[start : end + 1] == class_id)[0])
                    cx, cy, bw, bh = (float(value) for value in boxes_1024[start + local])
                    width, height = int(image["width"]), int(image["height"])
                    scale = max(width, height) / 1024.0
                    box = _clip_box_to_image(
                        (
                            (cx - bw / 2) * scale,
                            (cy - bh / 2) * scale,
                            (cx + bw / 2) * scale,
                            (cy + bh / 2) * scale,
                        ),
                        width,
                        height,
                    )
                    entry.update(object_index=local, box_xyxy=[round(value, 2) for value in box])
                sampled[class_id].append(entry)
        if all(len(items) >= per_class for items in sampled.values()):
            break
    return sampled


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("data/raw/vg150"))
    p.add_argument("--output-dir", type=Path, default=Path("data/knowledge/level1-human-review-v2"))
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--per-class", type=int, default=5)
    args = p.parse_args(argv)
    loader = VG150AnnotationLoader(
        root=args.root, split="train", split_manifest=args.root / "splits.json", box_policy="clip"
    )
    classes, vocab_hash = _read_classes(args.root / "VG-SGG-dicts-with-attri.json", 150)
    with h5py.File(args.root / "VG-SGG-with-attri.h5", "r") as h5:
        first = np.asarray(h5["img_to_first_box"][:]).reshape(-1)
        last = np.asarray(h5["img_to_last_box"][:]).reshape(-1)
        labels = np.asarray(h5["labels"][:]).reshape(-1)
        boxes = np.asarray(h5["boxes_1024"][:])
    sampled = sample_examples(
        loader.indices,
        first,
        last,
        labels,
        loader.image_metadata,
        boxes_1024=boxes,
        seed=args.seed,
        per_class=args.per_class,
    )
    stream = io.StringIO(newline="")
    boxes_stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    box_writer = csv.writer(boxes_stream, lineterminator="\n")
    writer.writerow(("class_id", "label", "train_example_count", "train_image_ids", "train_image_urls"))
    box_writer.writerow(("class_id", "label", "image_id", "object_index", "box_xyxy", "url"))
    for class_id, name in classes:
        examples = sampled[class_id]
        writer.writerow(
            (
                class_id,
                name,
                len(examples),
                ";".join(str(item["image_id"]) for item in examples),
                ";".join(item["url"] or "" for item in examples),
            )
        )
        for item in examples:
            box_writer.writerow(
                (class_id, name, item["image_id"], item["object_index"], json.dumps(item["box_xyxy"]), item["url"])
            )
    provenance = {
        "schema_version": 1,
        "split": "train_only",
        "source": "VG150 object labels and aligned image metadata only; no predicate annotations",
        "seed": args.seed,
        "per_class": args.per_class,
        "split_sha256": sha256(args.root / "splits.json"),
        "h5_sha256": sha256(args.root / "VG-SGG-with-attri.h5"),
        "vocabulary_sha256": vocab_hash,
        "classes_with_full_sample": sum(len(items) == args.per_class for items in sampled.values()),
    }
    box_provenance = {
        **provenance,
        "source": "VG150 train object labels, boxes_1024, and aligned image metadata; no predicate annotations",
        "box_policy": "clip",
        "box_encoding": "pixel_xyxy_first_matching_object_instance",
    }
    _save_outputs(
        args.output_dir,
        {
            "train_examples.csv": stream.getvalue().encode("utf-8-sig"),
            "train_examples_manifest.json": (json.dumps(provenance, indent=2) + "\n").encode("utf-8"),
            "train_examples_with_boxes.csv": boxes_stream.getvalue().encode("utf-8-sig"),
            "train_examples_with_boxes_manifest.json": (json.dumps(box_provenance, indent=2) + "\n").encode("utf-8"),
        },
        dry_run=False,
    )
    print(f"[TRAIN ONLY] {provenance['classes_with_full_sample']}/150 classes have {args.per_class} examples")


if __name__ == "__main__":
    main()
