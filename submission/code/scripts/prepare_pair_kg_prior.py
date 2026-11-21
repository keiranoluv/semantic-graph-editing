"""Count sampled training targets for a fixed logit-adjustment prior (CPU only)."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import h5py

from src.knowledge.transe.core import sha256

DATA_FILES = ("VG-SGG-with-attri.h5", "VG-SGG-dicts-with-attri.json", "image_data.json", "splits.json")


def make_prior(root: Path, train_limit: int, negative_ratio: int) -> dict:
    if train_limit <= 0 or negative_ratio < 0:
        raise ValueError("Invalid train limit or negative ratio")
    splits = json.loads((root / "splits.json").read_text(encoding="utf-8"))
    rows = splits["train"][:train_limit]
    vocab = json.loads((root / "VG-SGG-dicts-with-attri.json").read_text(encoding="utf-8"))
    num_classes = max(vocab["predicate_to_idx"].values()) + 1
    counts = Counter()
    skipped = 0
    with h5py.File(root / "VG-SGG-with-attri.h5", "r") as h5:
        for row in rows:
            first_box = int(h5["img_to_first_box"][row])
            last_box = int(h5["img_to_last_box"][row])
            first_rel = int(h5["img_to_first_rel"][row])
            last_rel = int(h5["img_to_last_rel"][row])
            n = last_box - first_box + 1
            by_pair = defaultdict(set)
            if first_rel >= 0:
                pairs = h5["relationships"][first_rel : last_rel + 1]
                labels = h5["predicates"][first_rel : last_rel + 1].reshape(-1)
                for pair, predicate in zip(pairs, labels, strict=True):
                    subject, object_ = int(pair[0]) - first_box, int(pair[1]) - first_box
                    if subject != object_:
                        by_pair[(subject, object_)].add(int(predicate))
            positives = [next(iter(labels)) for labels in by_pair.values() if len(labels) == 1]
            if not positives:
                skipped += 1
                continue
            counts.update(positives)
            backgrounds = n * (n - 1) - len(by_pair)
            counts[0] += min(backgrounds, negative_ratio * len(positives))
    raw = [counts[i] for i in range(num_classes)]
    smoothed = [count + 1 for count in raw]
    denominator = sum(smoothed)
    return {
        "schema_version": 1,
        "split": "train",
        "train_limit": len(rows),
        "negative_ratio": negative_ratio,
        "skipped_images": skipped,
        "class_counts": raw,
        "class_prior": [count / denominator for count in smoothed],
        "smoothing": "add_one",
        "sampling": "all_unambiguous_positives_plus_min(background_pairs,ratio*positive_pairs)",
        "data_sha256": {name: sha256(root / name) for name in DATA_FILES},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/vg150"))
    parser.add_argument("--train-limit", type=int, default=57723)
    parser.add_argument("--negative-ratio", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("reports/level1/clip-pair-kg-train-prior-v1.json"))
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    result = make_prior(args.root, args.train_limit, args.negative_ratio)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(f"[PRIOR] train={result['train_limit']} skipped={result['skipped_images']} output={args.output}")


if __name__ == "__main__":
    main()
