"""Evaluate one saved GAT checkpoint on VG150 PredCls validation."""

import argparse
import json
from pathlib import Path

import torch
from tqdm.auto import tqdm

from scripts.pilot_mlp import resolve_device
from scripts.train_mlp import make_dataset
from src.evaluation.gat import evaluate_gat_batch
from src.evaluation.scene_graph import summarize_evaluation
from src.models.gat import build_gat_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("data/raw/vg150"))
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--allow-test", action="store_true", help="Explicitly unlock held-out final test evaluation")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-images", type=int, default=None, help="Defaults to the entire selected split")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.split == "test" and not args.allow_test:
        parser.error("Test split is reserved for final evaluation; pass --allow-test explicitly")
    if args.batch_size <= 0 or (args.max_images is not None and args.max_images <= 0):
        parser.error("Batch size and optional image limit must be positive")

    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("checkpoint_version") != 3:
        raise ValueError("Expected a GAT checkpoint with version 3")
    config = checkpoint["config"]
    model = build_gat_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    dataset = make_dataset(args.root, args.split, config["box_policy"])
    limit = len(dataset) if args.max_images is None else min(args.max_images, len(dataset))
    results = []

    for start in tqdm(range(0, limit, args.batch_size), desc="Evaluate GAT", unit="batch"):
        graphs = [dataset[i] for i in range(start, min(start + args.batch_size, limit))]
        batch_results, _, _ = evaluate_gat_batch(model, graphs, device)
        results.extend(batch_results)

    report = summarize_evaluation(results)
    report.update(
        {
            "dataset": "VG150",
            "model": "gat",
            "split": args.split,
            "images_evaluated": limit,
            "checkpoint": str(args.checkpoint),
            "checkpoint_epoch": checkpoint["epoch"],
            "seed": config["seed"],
            "box_policy": config["box_policy"],
            "protocol": "PredCls_graph_constrained_top1_per_pair",
            "recall_aggregation": "image_macro",
            "mean_recall_aggregation": "predicate_macro",
            "git_sha": checkpoint.get("git_sha"),
            "inference_batch_size": args.batch_size,
            "device": str(device),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
