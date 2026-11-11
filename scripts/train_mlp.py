"""Train VG150 geometry-only MLP with optional W&B tracking."""

import argparse
import math
import subprocess
from pathlib import Path

import torch
from torch.nn import functional as F

import wandb
from src.datasets.vg150 import VG150AnnotationLoader, VG150GraphDataset
from src.features.geometry import GeometryFeatureExtractor
from src.features.visual import GeometryVisualFeatureExtractor, VisualFeatureExtractor
from src.graph.builder import SceneGraphBuilder
from src.models.mlp import MLPPredicateClassifier
from src.training.trainer import train_step


def make_dataset(root, split, box_policy, visual_cache=None):
    loader = VG150AnnotationLoader(
        root=root,
        split=split,
        split_manifest=root / "splits.json",
        box_policy=box_policy,
    )
    extractor = (
        GeometryVisualFeatureExtractor(VisualFeatureExtractor(visual_cache, root, box_policy))
        if visual_cache is not None
        else GeometryFeatureExtractor()
    )
    return VG150GraphDataset(
        loader=loader,
        feature_extractor=extractor,
        graph_builder=SceneGraphBuilder(),
    )


@torch.no_grad()
def validate(model, dataset, limit):
    """Evaluate cross-entropy over all non-ambiguous candidate pairs."""
    model.eval()
    total_loss = 0.0
    total_pairs = 0

    for idx in range(limit):
        graph = dataset[idx]
        mask = graph.edge_label >= 0

        if not mask.any():
            continue

        logits = model(graph.x, graph.edge_label_index[:, mask])
        targets = graph.edge_label[mask]

        loss = F.cross_entropy(logits, targets, reduction="sum")
        total_loss += loss.item()
        total_pairs += targets.numel()

    return total_loss / total_pairs if total_pairs else float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/vg150"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-train-images", type=int, default=100)
    parser.add_argument("--max-val-images", type=int, default=20)
    parser.add_argument("--box-policy", choices=["strict", "clip"], default="clip")
    parser.add_argument("--negative-ratio", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/mlp_latest.pt"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default="offline",
    )
    args = parser.parse_args()

    if args.epochs <= 0 or args.max_train_images <= 0 or args.max_val_images <= 0:
        parser.error("Epoch and image limits must be positive")
    if args.negative_ratio < 0 or args.lr <= 0:
        parser.error("Invalid negative ratio or learning rate")

    torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    train_data = make_dataset(args.root, "train", args.box_policy)
    val_data = make_dataset(args.root, "val", args.box_policy)

    train_limit = min(args.max_train_images, len(train_data))
    val_limit = min(args.max_val_images, len(val_data))

    model = MLPPredicateClassifier(
        input_dim=5,
        hidden_dim=256,
        num_classes=51,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    generator = torch.Generator().manual_seed(args.seed)

    config = {
        "dataset": "VG150",
        "root": str(args.root.resolve()),
        "model": "mlp",
        "features": "geometry",
        "hidden_dim": 256,
        "num_classes": 51,
        "box_policy": args.box_policy,
        "negative_ratio": args.negative_ratio,
        "lr": args.lr,
        "seed": args.seed,
        "train_limit": train_limit,
        "val_limit": val_limit,
    }

    start_epoch = 0
    run_id = None

    if args.resume:
        checkpoint = torch.load(
            args.checkpoint,
            map_location="cpu",
            weights_only=True,
        )

        if checkpoint["config"] != config:
            raise ValueError("Checkpoint configuration mismatch")

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        generator.set_state(checkpoint["sampler_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])

        start_epoch = checkpoint["epoch"]
        run_id = checkpoint.get("wandb_run_id")

        print(f"[RESUME] From epoch {start_epoch}", flush=True)

    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()

    run = None

    if args.wandb_mode != "disabled":
        run = wandb.init(
            project="semantic-graph-editing",
            name=f"mlp-geometry-seed{args.seed}",
            id=run_id,
            resume="allow" if run_id else "never",
            mode=args.wandb_mode,
            config={**config, "git_sha": git_sha},
        )

    print(f"[INFO] Train images: {train_limit}", flush=True)
    print(f"[INFO] Validation images: {val_limit}", flush=True)

    try:
        for epoch in range(start_epoch, args.epochs):
            losses = []
            skipped = 0

            for idx in range(train_limit):
                graph = train_data[idx]

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

                losses.append(loss)

            if not losses:
                raise RuntimeError("No successful training steps")

            train_loss = sum(losses) / len(losses)
            val_loss = validate(model, val_data, val_limit)

            if not math.isfinite(val_loss):
                raise RuntimeError("Non-finite validation loss")

            metrics = {
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "train/optimizer_steps": len(losses),
                "train/skipped_graphs": skipped,
                "train/learning_rate": optimizer.param_groups[0]["lr"],
            }

            print(
                f"[Epoch {epoch + 1}/{args.epochs}] "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} "
                f"steps={len(losses)} "
                f"skipped={skipped}",
                flush=True,
            )

            if run is not None:
                run.log(metrics, step=epoch + 1)

            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "sampler_rng": generator.get_state(),
                    "torch_rng": torch.get_rng_state(),
                    "config": config,
                    "wandb_run_id": run.id if run is not None else None,
                },
                args.checkpoint,
            )

        print("[PASS] Training completed", flush=True)
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
