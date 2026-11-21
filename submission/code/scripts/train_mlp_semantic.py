"""Train semantic MLP on VG150 with optional W&B tracking."""

import argparse
import json
import math
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F

from scripts.train_mlp import make_dataset
from src.models.mlp_semantic import SemanticMLPPredicateClassifier
from src.training.sampling import sample_training_pairs


def forward(model, graph, pairs):
    return model(
        x=graph.x,
        edge_label_index=pairs,
        object_labels=graph.object_labels,
    )


def train_step(model, graph, optimizer, ratio, generator):
    indices = sample_training_pairs(
        graph.edge_label,
        negative_ratio=ratio,
        generator=generator,
    )
    if indices.numel() == 0:
        return None

    model.train()
    optimizer.zero_grad()

    logits = forward(model, graph, graph.edge_label_index[:, indices])
    loss = F.cross_entropy(logits, graph.edge_label[indices])

    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite training loss")

    loss.backward()
    optimizer.step()
    return float(loss.item())


@torch.no_grad()
def validate(model, dataset, limit):
    model.eval()
    total_loss = 0.0
    total_pairs = 0

    for i in range(limit):
        graph = dataset[i]
        mask = graph.edge_label >= 0

        if not mask.any():
            continue

        logits = forward(model, graph, graph.edge_label_index[:, mask])
        labels = graph.edge_label[mask]

        total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
        total_pairs += labels.numel()

    if total_pairs == 0:
        raise RuntimeError("No validation candidate pairs")

    return total_loss / total_pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/vg150"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-train-images", type=int, default=100)
    parser.add_argument("--max-val-images", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--negative-ratio", type=int, default=3)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--box-policy", choices=["strict", "clip"], default="clip")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/mlp_semantic.pt"),
    )
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default="offline",
    )
    args = parser.parse_args()

    if (
        min(
            args.epochs,
            args.max_train_images,
            args.max_val_images,
            args.embedding_dim,
            args.hidden_dim,
        )
        <= 0
    ):
        parser.error("Epoch, image limits and dimensions must be positive")
    if args.negative_ratio < 0 or args.lr <= 0:
        parser.error("Invalid negative ratio or learning rate")

    torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    with (args.root / "VG-SGG-dicts-with-attri.json").open() as f:
        vocabulary = json.load(f)

    num_objects = max(vocabulary["label_to_idx"].values()) + 1
    num_predicates = max(vocabulary["predicate_to_idx"].values()) + 1

    if num_predicates != 51:
        raise ValueError(f"Expected 51 predicate classes, got {num_predicates}")

    train_data = make_dataset(args.root, "train", args.box_policy)
    val_data = make_dataset(args.root, "val", args.box_policy)

    train_limit = min(args.max_train_images, len(train_data))
    val_limit = min(args.max_val_images, len(val_data))

    model = SemanticMLPPredicateClassifier(
        geometry_dim=5,
        num_object_classes=num_objects,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_classes=num_predicates,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    generator = torch.Generator().manual_seed(args.seed)

    config = {
        "model": "mlp_semantic",
        "dataset": "VG150",
        "features": "geometry+trainable_object_embedding",
        "seed": args.seed,
        "lr": args.lr,
        "negative_ratio": args.negative_ratio,
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "num_object_classes": num_objects,
        "num_classes": num_predicates,
        "box_policy": args.box_policy,
        "train_limit": train_limit,
        "val_limit": val_limit,
        "root": str(args.root.resolve()),
    }

    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()

    run = None
    if args.wandb_mode != "disabled":
        import wandb

        run = wandb.init(
            project="semantic-graph-editing",
            name=f"mlp-semantic-seed{args.seed}",
            config={**config, "git_sha": sha},
            mode=args.wandb_mode,
        )

    print(f"[INFO] Object embedding rows: {num_objects}")
    print(f"[INFO] Predicate classes: {num_predicates}")
    print(f"[INFO] Training images: {train_limit}")
    print(f"[INFO] Validation images: {val_limit}")

    try:
        for epoch in range(args.epochs):
            losses = []
            skipped = 0

            for idx in range(train_limit):
                graph = train_data[idx]
                loss = train_step(
                    model,
                    graph,
                    optimizer,
                    args.negative_ratio,
                    generator,
                )
                if loss is None:
                    skipped += 1
                else:
                    losses.append(loss)

            if not losses:
                raise RuntimeError("No successful optimizer steps")

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
            }

            print(
                f"[Epoch {epoch + 1}/{args.epochs}] "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} "
                f"steps={len(losses)} skipped={skipped}",
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
                    "config": config,
                },
                args.checkpoint,
            )

        print("[PASS] Semantic MLP training completed")
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
