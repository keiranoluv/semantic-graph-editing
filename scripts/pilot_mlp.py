"""Pilot training for geometry and semantic VG150 PredCls MLPs."""

import argparse
import json
import math
import random
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F

from scripts.evaluate_mlp import build_model, predict
from scripts.train_mlp import make_dataset
from src.evaluation.scene_graph import evaluate_image, summarize_evaluation
from src.training.sampling import sample_training_pairs


def train_one_epoch(model, dataset, limit, optimizer, ratio, generator, name):
    model.train()
    losses = []
    skipped = 0

    for idx in range(limit):
        graph = dataset[idx]
        indices = sample_training_pairs(
            graph.edge_label,
            negative_ratio=ratio,
            generator=generator,
        )

        if indices.numel() == 0:
            skipped += 1
            continue

        optimizer.zero_grad(set_to_none=True)

        # Reuse the existing evaluation dispatch to handle both models.
        # Restrict candidate pairs before the forward pass.
        original_pairs = graph.edge_label_index
        graph.edge_label_index = original_pairs[:, indices]

        try:
            logits = predict(model, graph, name)
        finally:
            graph.edge_label_index = original_pairs

        loss = F.cross_entropy(logits, graph.edge_label[indices])
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite train loss at image {idx}")

        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    if not losses:
        raise RuntimeError("No successful optimizer steps")

    return sum(losses) / len(losses), len(losses), skipped


@torch.no_grad()
def validate(model, dataset, limit, name):
    model.eval()
    total_loss = 0.0
    total_pairs = 0
    results = []

    for idx in range(limit):
        graph = dataset[idx]
        logits = predict(model, graph, name)

        # Cross-entropy: all non-ambiguous candidate pairs.
        mask = graph.edge_label >= 0
        if mask.any():
            total_loss += F.cross_entropy(
                logits[mask],
                graph.edge_label[mask],
                reduction="sum",
            ).item()
            total_pairs += int(mask.sum().item())

        # PredCls: GT triplets are evaluated independently
        # of the single-label training targets.
        results.append(
            evaluate_image(
                logits=logits,
                candidate_pairs=graph.edge_label_index,
                gt_pairs=graph.gt_relation_index,
                gt_predicates=graph.gt_predicate,
            )
        )

    if total_pairs == 0:
        raise RuntimeError("No validation candidate pairs")

    metrics = summarize_evaluation(results)
    metrics["loss"] = total_loss / total_pairs
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=["mlp", "mlp_semantic"],
        required=True,
    )
    parser.add_argument("--root", type=Path, default=Path("data/raw/vg150"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--negative-ratio", type=int, default=3)
    parser.add_argument("--max-train-images", type=int, default=57723)
    parser.add_argument("--max-val-images", type=int, default=5000)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--box-policy", choices=["strict", "clip"], default="clip")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/pilot"))
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default="offline",
    )
    args = parser.parse_args()

    if (
        min(
            args.epochs,
            args.patience,
            args.max_train_images,
            args.max_val_images,
            args.hidden_dim,
            args.embedding_dim,
        )
        <= 0
    ):
        parser.error("Epochs, patience, image limits and dimensions must be positive")

    if args.negative_ratio < 0 or args.lr <= 0:
        parser.error("Invalid negative ratio or learning rate")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    vocab_path = args.root / "VG-SGG-dicts-with-attri.json"
    with vocab_path.open(encoding="utf-8") as f:
        vocab = json.load(f)

    num_objects = max(vocab["label_to_idx"].values()) + 1
    num_predicates = max(vocab["predicate_to_idx"].values()) + 1

    if num_predicates != 51:
        raise ValueError(f"Expected 51 predicate classes, got {num_predicates}")

    train_data = make_dataset(args.root, "train", args.box_policy)
    val_data = make_dataset(args.root, "val", args.box_policy)

    train_limit = min(args.max_train_images, len(train_data))
    val_limit = min(args.max_val_images, len(val_data))

    config = {
        "model": args.model,
        "dataset": "VG150",
        "features": ("geometry" if args.model == "mlp" else "geometry+trainable_object_embedding"),
        "hidden_dim": args.hidden_dim,
        "embedding_dim": args.embedding_dim,
        "num_object_classes": num_objects,
        "num_classes": num_predicates,
        "box_policy": args.box_policy,
        "negative_ratio": args.negative_ratio,
        "lr": args.lr,
        "seed": args.seed,
        "train_limit": train_limit,
        "val_limit": val_limit,
        "selection_metric": "mR@50",
    }

    model = build_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    generator = torch.Generator().manual_seed(args.seed)

    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()

    run_dir = args.output_dir / f"{args.model}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    run = None
    if args.wandb_mode != "disabled":
        import wandb

        run = wandb.init(
            project="semantic-graph-editing",
            name=f"pilot-{args.model}-seed{args.seed}",
            mode=args.wandb_mode,
            config={**config, "git_sha": git_sha},
        )

    best_score = -math.inf
    best_epoch = 0
    bad_epochs = 0
    history = []

    print(f"[INFO] Model: {args.model}", flush=True)
    print(f"[INFO] Train images: {train_limit}", flush=True)
    print(f"[INFO] Validation images: {val_limit}", flush=True)

    try:
        for epoch in range(1, args.epochs + 1):
            train_loss, steps, skipped = train_one_epoch(
                model,
                train_data,
                train_limit,
                optimizer,
                args.negative_ratio,
                generator,
                args.model,
            )

            val = validate(model, val_data, val_limit, args.model)
            score = val["mR@50"]

            if score is None or not math.isfinite(score):
                raise RuntimeError("Invalid mR@50; cannot select checkpoint")

            metrics = {
                "epoch": epoch,
                "train/loss": train_loss,
                "train/optimizer_steps": steps,
                "train/skipped_graphs": skipped,
                "val/loss": val["loss"],
                "val/R@50": val["R@50"],
                "val/R@100": val["R@100"],
                "val/mR@50": val["mR@50"],
                "val/mR@100": val["mR@100"],
            }
            history.append(metrics)

            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "sampler_rng": generator.get_state(),
                "torch_rng": torch.get_rng_state(),
                "config": config,
                "validation": val,
                "git_sha": git_sha,
            }

            torch.save(checkpoint, run_dir / "last.pt")

            if score > best_score:
                best_score = score
                best_epoch = epoch
                bad_epochs = 0
                torch.save(checkpoint, run_dir / "best.pt")
                improved = True
            else:
                bad_epochs += 1
                improved = False

            (run_dir / "history.json").write_text(
                json.dumps(history, indent=2) + "\n",
                encoding="utf-8",
            )

            print(
                f"[Epoch {epoch:02d}/{args.epochs}] "
                f"train={train_loss:.4f} "
                f"val_loss={val['loss']:.4f} "
                f"R50={val['R@50']:.4f} "
                f"mR50={score:.4f} "
                f"best={best_score:.4f}@{best_epoch} "
                f"{'NEW_BEST' if improved else ''}",
                flush=True,
            )

            if run is not None:
                run.log(metrics, step=epoch)

            if bad_epochs >= args.patience:
                print(
                    f"[EARLY STOP] No mR@50 improvement " f"for {args.patience} epochs",
                    flush=True,
                )
                break

        print(
            f"[PASS] Best epoch={best_epoch}, " f"best mR@50={best_score:.4f}",
            flush=True,
        )
        print(f"[INFO] Checkpoints: {run_dir}", flush=True)

    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
