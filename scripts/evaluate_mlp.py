"""Evaluate an MLP checkpoint on VG150 PredCls."""

import argparse
import json
from pathlib import Path

import torch

from src.datasets.vg150 import VG150AnnotationLoader, VG150GraphDataset
from src.evaluation.scene_graph import evaluate_image, summarize_evaluation
from src.features.geometry import GeometryFeatureExtractor
from src.graph.builder import SceneGraphBuilder
from src.models.mlp import MLPPredicateClassifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/mlp_wandb_smoke.pt"),
    )
    parser.add_argument("--root", type=Path, default=Path("data/raw/vg150"))
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/mlp_evaluation.json"),
    )
    args = parser.parse_args()

    if args.max_images <= 0:
        parser.error("--max-images must be positive")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    box_policy = config["box_policy"]

    loader = VG150AnnotationLoader(
        root=args.root,
        split=args.split,
        split_manifest=args.root / "splits.json",
        box_policy=box_policy,
    )

    extractor = GeometryFeatureExtractor()
    dataset = VG150GraphDataset(
        loader=loader,
        feature_extractor=extractor,
        graph_builder=SceneGraphBuilder(),
    )

    model = MLPPredicateClassifier(
        input_dim=extractor.output_dim,
        hidden_dim=config["hidden_dim"],
        num_classes=config["num_classes"],
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()

    results = []
    limit = min(args.max_images, len(dataset))

    with torch.no_grad():
        for idx in range(limit):
            graph = dataset[idx]
            logits = model(graph.x, graph.edge_label_index)

            result = evaluate_image(
                logits=logits,
                candidate_pairs=graph.edge_label_index,
                gt_pairs=graph.gt_relation_index,
                gt_predicates=graph.gt_predicate,
            )
            results.append(result)

    summary = summarize_evaluation(results)
    summary.update(
        {
            "dataset": "VG150",
            "split": args.split,
            "images_evaluated": limit,
            "box_policy": box_policy,
            "checkpoint": str(args.checkpoint),
            "checkpoint_epoch": checkpoint["epoch"],
            "protocol": "PredCls_graph_constrained_top1_per_pair",
            "recall_aggregation": "image_macro",
            "mean_recall_aggregation": "predicate_macro",
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
