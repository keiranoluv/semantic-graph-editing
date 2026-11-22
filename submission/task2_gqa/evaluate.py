"""Evaluate one GQA checkpoint on an existing feature cache."""

import argparse
import json
from pathlib import Path

from common.io import read, sha256
from task2_gqa.cache import GQAFullDataset
from task2_gqa.dataset import HeldoutDataset
from task2_gqa.predict import MANIFEST, build
from task2_gqa.validation import evaluate, evaluate_fast


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--backend", choices=("base", "fast", "ablation", "kg"), default="fast")
    p.add_argument("--cache", type=Path, default=Path("data/cache/gqa-full-clip-v1"))
    p.add_argument("--split", choices=("train", "val", "test"), default="val")
    p.add_argument("--allow-test", action="store_true")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-images", type=int)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    if a.split == "test" and not a.allow_test:
        p.error("--split test requires --allow-test after model selection")
    if a.batch_size < 1 or (a.max_images is not None and a.max_images < 1):
        p.error("Batch size and image limit must be positive")
    model, dims = build(
        dict(checkpoint=str(a.checkpoint), checkpoint_sha256=sha256(a.checkpoint), model=a.model, backend=a.backend)
    )
    metadata = read(a.cache / "manifest.json")
    if metadata["identity"]["manifest_sha256"] != sha256(MANIFEST):
        raise ValueError("Cache and GQA split manifest differ")
    data = HeldoutDataset(a.cache, metadata) if a.split == "test" else GQAFullDataset(a.cache, metadata, a.split)
    evaluator = evaluate_fast if a.backend in ("fast", "ablation") else evaluate
    try:
        count = len(data) if a.max_images is None else min(len(data), a.max_images)
        graphs = [data[i] for i in range(count)]
        result = evaluator(
            model.to(a.device),
            "gat" if a.backend == "kg" else a.model,
            graphs,
            {**dims, "batch_size": a.batch_size},
            a.device,
            include_ng=True,
        )
        text = json.dumps({"images": count, "metrics": result}, indent=2, allow_nan=False)
        if a.output:
            a.output.parent.mkdir(parents=True, exist_ok=True)
            with a.output.open("x") as stream:
                stream.write(text + "\n")
        print(text)
    finally:
        data.close()


if __name__ == "__main__":
    main()
