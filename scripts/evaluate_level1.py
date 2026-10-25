"""Shared frozen-checkpoint evaluation for MLP, GAT and GAT+KG (val by default)."""

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from scripts.evaluate_mlp import build_model
from scripts.pilot_mlp import resolve_device
from scripts.train_mlp import make_dataset
from src.evaluation.batched import evaluate_graph_batch
from src.evaluation.gat import evaluate_gat_batch
from src.evaluation.scene_graph import summarize_evaluation
from src.features.visual import load_visual_manifest
from src.knowledge.transe.core import sha256
from src.models.gat import build_gat_model
from src.models.gat_kg import model_from_checkpoint


def load_model(checkpoint):
    """Reconstruct every level 1 baseline and the new KG model."""
    config = checkpoint["config"]
    name = config["model"]
    if name == "gat_kg":
        return model_from_checkpoint(checkpoint)
    if name == "gat":
        model = build_gat_model(config)
    else:
        model = build_model(config)
    model.load_state_dict(checkpoint["model"])
    return model


@torch.inference_mode()
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--root", type=Path, default=Path("data/raw/vg150"))
    p.add_argument("--visual-cache", type=Path, help="Override cache path stored in a visual checkpoint")
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--allow-test", action="store_true")
    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-images", type=int)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    if args.split == "test" and not args.allow_test:
        p.error("Held-out test requires --allow-test after experimental setup is frozen")
    if args.batch_size <= 0 or (args.max_images is not None and args.max_images <= 0):
        p.error("Batch size and optional image limit must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    config = checkpoint["config"]
    name = config["model"]
    data_hashes = {
        name: sha256(args.root / name)
        for name in ["VG-SGG-with-attri.h5", "VG-SGG-dicts-with-attri.json", "image_data.json", "splits.json"]
    }
    if config.get("data_sha256") is not None and config["data_sha256"] != data_hashes:
        raise ValueError("Evaluation data differs from training input hashes")
    visual_cache = args.visual_cache or config.get("visual_cache")
    if config.get("visual_cache_manifest_sha256"):
        if visual_cache is None:
            raise ValueError("Visual checkpoint requires its frozen region cache")
        visual_cache = Path(visual_cache)
        load_visual_manifest(visual_cache, args.root, config["box_policy"])
        if sha256(visual_cache / "manifest.json") != config["visual_cache_manifest_sha256"]:
            raise ValueError("Visual cache differs from training checkpoint")
    elif args.visual_cache is not None:
        raise ValueError("Cannot add visual features to an annotation-only checkpoint")
    model = load_model(checkpoint)
    model.to(device).eval()
    dataset = make_dataset(args.root, args.split, config["box_policy"], visual_cache)
    limit = len(dataset) if args.max_images is None else min(args.max_images, len(dataset))
    results = []
    for begin in range(0, limit, args.batch_size):
        graphs = [dataset[i] for i in range(begin, min(begin + args.batch_size, limit))]
        if name in {"gat", "gat_kg"}:
            batch_results, _, _ = evaluate_gat_batch(model, graphs, device)
        else:
            batch_results, _, _ = evaluate_graph_batch(model, graphs, name, device)
        results.extend(batch_results)
    support = Counter()
    hits = {50: Counter(), 100: Counter()}
    for result in results:
        support.update(result["class_support"])
        for k in hits:
            hits[k].update(result["class_hits"][k])
    report = {
        **summarize_evaluation(results),
        "model": name,
        "seed": config["seed"],
        "split": args.split,
        "images_evaluated": limit,
        "split_images": len(dataset),
        "full_split": limit == len(dataset),
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint": str(args.checkpoint),
        "config": config,
        "data_sha256": data_hashes,
        "historical_training_data_hashes_available": "data_sha256" in config,
        "protocol": "PredCls_graph_constrained_top1_per_pair",
        "recall_aggregation": "image_macro",
        "mean_recall_aggregation": "predicate_macro_over_gt_support",
        "interpretation": config.get("interpretation", "baseline_evaluation"),
        "per_predicate": {
            str(c): {"support": support[c], **{f"R@{k}": hits[k][c] / support[c] for k in hits}}
            for c in sorted(support)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ["model", "seed", "images_evaluated", "R@50", "mR@50"]}, indent=2))


if __name__ == "__main__":
    main()
