"""Evaluate geometry or semantic MLP on VG150 PredCls."""

import argparse
import json
from pathlib import Path

import torch
from tqdm.auto import tqdm

from src.datasets.vg150 import VG150AnnotationLoader, VG150GraphDataset
from src.evaluation.batched import evaluate_graph_batch
from src.evaluation.scene_graph import summarize_evaluation
from src.features.geometry import GeometryFeatureExtractor
from src.graph.builder import SceneGraphBuilder
from src.models.mlp import MLPPredicateClassifier
from src.models.mlp_semantic import SemanticMLPPredicateClassifier


def build_model(config):
    """Reconstruct the model architecture from checkpoint configuration."""
    model_name = config["model"]

    if model_name in ("mlp", "mlp_geometry"):
        return MLPPredicateClassifier(
            input_dim=config.get("geometry_dim", 5),
            hidden_dim=config["hidden_dim"],
            num_classes=config["num_classes"],
        )

    if model_name == "mlp_semantic":
        return SemanticMLPPredicateClassifier(
            geometry_dim=config.get("geometry_dim", 5),
            num_object_classes=config["num_object_classes"],
            embedding_dim=config["embedding_dim"],
            hidden_dim=config["hidden_dim"],
            num_classes=config["num_classes"],
        )

    raise ValueError(f"Unsupported model: {model_name}")


def predict(model, graph, model_name):
    """Run the appropriate forward interface."""
    if model_name == "mlp_semantic":
        return model(
            x=graph.x,
            edge_label_index=graph.edge_label_index,
            object_labels=graph.object_labels,
        )

    return model(
        x=graph.x,
        edge_label_index=graph.edge_label_index,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("data/raw/vg150"))
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.batch_size <= 0 or args.max_images <= 0:
        parser.error("Batch size and max images must be positive")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    model_name = config["model"]
    box_policy = config["box_policy"]

    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    loader = VG150AnnotationLoader(
        root=args.root,
        split=args.split,
        split_manifest=args.root / "splits.json",
        box_policy=box_policy,
    )
    dataset = VG150GraphDataset(
        loader=loader,
        feature_extractor=GeometryFeatureExtractor(),
        graph_builder=SceneGraphBuilder(),
    )

    limit = min(args.max_images, len(dataset))
    results = []

    with torch.inference_mode():
        progress = tqdm(
            range(0, limit, args.batch_size),
            desc="Evaluate",
            unit="batch",
        )

        for begin in progress:
            end = min(begin + args.batch_size, limit)
            graphs = [dataset[idx] for idx in range(begin, end)]

            batch_results, _, _ = evaluate_graph_batch(model, graphs, model_name, device)
            results.extend(batch_results)
            progress.set_postfix(images=end)

    summary = summarize_evaluation(results)
    summary.update(
        {
            "dataset": "VG150",
            "model": model_name,
            "features": config["features"],
            "split": args.split,
            "images_evaluated": limit,
            "box_policy": box_policy,
            "checkpoint": str(args.checkpoint),
            "checkpoint_epoch": checkpoint["epoch"],
            "seed": config.get("seed"),
            "device": str(device),
            "inference_batch_size": args.batch_size,
            "protocol": "PredCls_graph_constrained_top1_per_pair",
            "recall_aggregation": "image_macro",
            "mean_recall_aggregation": "predicate_macro",
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
