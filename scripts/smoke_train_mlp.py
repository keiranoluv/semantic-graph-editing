"""Smoke-test MLP training with real VG150 annotations."""

import argparse
import math
from pathlib import Path

import torch

from src.datasets.vg150 import VG150AnnotationLoader, VG150GraphDataset
from src.features.geometry import GeometryFeatureExtractor
from src.graph.builder import SceneGraphBuilder
from src.models.mlp import MLPPredicateClassifier
from src.training.trainer import train_step


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/vg150"))
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--box-policy", choices=("strict", "clip"), default="clip")
    parser.add_argument("--negative-ratio", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.steps <= 0:
        parser.error("--steps must be positive")

    torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    loader = VG150AnnotationLoader(
        root=args.root,
        split="train",
        split_manifest=args.root / "splits.json",
        box_policy=args.box_policy,
    )

    extractor = GeometryFeatureExtractor()
    dataset = VG150GraphDataset(
        loader=loader,
        feature_extractor=extractor,
        graph_builder=SceneGraphBuilder(),
    )

    model = MLPPredicateClassifier(
        input_dim=extractor.output_dim,
        hidden_dim=256,
        num_classes=51,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
    )

    generator = torch.Generator().manual_seed(args.seed)
    successful_steps = 0
    skipped = 0
    losses = []

    print(f"[INFO] Train images: {len(dataset)}", flush=True)
    print(f"[INFO] Box policy: {args.box_policy}", flush=True)

    for idx in range(len(dataset)):
        graph = dataset[idx]

        positives = int((graph.edge_label > 0).sum())
        backgrounds = int((graph.edge_label == 0).sum())
        ambiguous = int((graph.edge_label == -1).sum())

        if graph.x.ndim != 2 or graph.x.shape[1] != 5:
            raise ValueError("Unexpected node features")

        if graph.edge_label_index.shape[0] != 2:
            raise ValueError("Unexpected candidate shape")

        valid = graph.edge_label[graph.edge_label >= 0]
        if valid.numel() and int(valid.max()) >= 51:
            raise ValueError("Predicate label outside class range")

        loss = train_step(
            model=model,
            graph=graph,
            optimizer=optimizer,
            negative_ratio=args.negative_ratio,
            generator=generator,
        )

        if loss is None:
            skipped += 1
            continue

        if not math.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at image {idx}")

        successful_steps += 1
        losses.append(loss)

        print(
            f"[Step {successful_steps:02d}/{args.steps}] "
            f"image={graph.image_id} "
            f"nodes={graph.num_nodes} "
            f"positive={positives} "
            f"background={backgrounds} "
            f"ambiguous={ambiguous} "
            f"loss={loss:.4f}",
            flush=True,
        )

        if successful_steps >= args.steps:
            break

    if successful_steps < args.steps:
        raise RuntimeError(f"Only {successful_steps}/{args.steps} training steps completed")

    print(f"\n[PASS] Completed {successful_steps} optimizer steps " f"(skipped {skipped} images)")
    print(f"[INFO] Mean loss: {sum(losses) / len(losses):.4f}")


if __name__ == "__main__":
    main()
