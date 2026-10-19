"""Train a label-independent GAT PredCls baseline on VG150 (train/val only)."""

import argparse
import json
import math
import os
import random
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from scripts.pilot_mlp import (
    atomic_text_save,
    atomic_torch_save,
    create_run_dir,
    resolve_device,
    restore_training_state,
)
from scripts.train_mlp import make_dataset
from src.evaluation.gat import validate_gat
from src.graph.gnn_batching import collate_gnn_graphs, forward_gat_batch
from src.models.gat import build_gat_model


def train_one_epoch(
    model, dataset, limit, batch_size, accumulation_steps, optimizer, negative_ratio, generator, device
):
    """Use graph mini-batches; supervise sampled pairs, propagate on full topology."""
    model.train()
    total_loss = 0.0
    total_pairs = 0
    optimizer_steps = 0
    skipped = 0
    pending_batches = 0
    pending_pairs = 0
    optimizer.zero_grad(set_to_none=True)

    def flush_gradients():
        """Normalize summed CE gradients by all supervised pairs in one step."""
        nonlocal optimizer_steps, pending_batches, pending_pairs
        if pending_batches == 0:
            return
        if pending_pairs <= 0:
            raise RuntimeError("No supervised pairs in accumulated step")
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(pending_pairs)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1
        pending_batches = 0
        pending_pairs = 0

    for begin in tqdm(range(0, limit, batch_size), desc="Train GAT", unit="micro-batch"):
        graphs = [dataset[i] for i in range(begin, min(begin + batch_size, limit))]
        batch, _, batch_skipped = collate_gnn_graphs(
            graphs, training=True, negative_ratio=negative_ratio, generator=generator
        )
        skipped += batch_skipped
        if batch is None:
            continue

        logits = forward_gat_batch(model, batch, device)
        targets = batch["targets"].to(device)
        loss_sum = F.cross_entropy(logits, targets, reduction="sum")
        if not torch.isfinite(loss_sum).item():
            raise RuntimeError(f"Non-finite GAT loss in micro-batch {begin}")
        loss_sum.backward()
        count = targets.numel()
        total_loss += float(loss_sum.detach().item())
        total_pairs += count
        pending_pairs += count
        pending_batches += 1
        if pending_batches == accumulation_steps:
            flush_gradients()

    flush_gradients()  # The final partial group must also update the model.
    if optimizer_steps == 0:
        raise RuntimeError("No successful optimizer steps")
    return total_loss / total_pairs, optimizer_steps, skipped


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/vg150"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32, help="Graphs per GPU micro-batch")
    parser.add_argument("--accumulation-steps", type=int, default=16, help="Micro-batches per optimizer update")
    parser.add_argument("--val-batch-size", type=int, default=32)
    parser.add_argument("--max-train-images", type=int, default=57723)
    parser.add_argument("--max-val-images", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--negative-ratio", type=int, default=3)
    parser.add_argument("--geometry-dim", type=int, default=5)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--box-policy", choices=("strict", "clip"), default="clip")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/experiments"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", type=Path, default=None, help="Resume from GAT last.pt")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="offline")
    args = parser.parse_args(argv)

    ints = (
        args.epochs,
        args.batch_size,
        args.accumulation_steps,
        args.val_batch_size,
        args.max_train_images,
        args.max_val_images,
        args.patience,
        args.geometry_dim,
        args.embedding_dim,
        args.hidden_dim,
        args.num_heads,
        args.num_layers,
    )
    if min(ints) <= 0 or args.geometry_dim != 5 or args.hidden_dim % args.num_heads:
        parser.error(
            "Dimensions, epochs, batch sizes and limits must be positive; 5D geometry and divisible heads required"
        )
    if args.lr <= 0 or args.weight_decay < 0 or args.negative_ratio < 0 or args.min_delta < 0:
        parser.error("Invalid optimizer, negative sampling or early-stopping parameters")
    if not 0 <= args.dropout < 1:
        parser.error("Dropout must be in [0, 1)")
    return args


def frozen_config(args, num_objects, num_predicates, train_limit, val_limit):
    """Everything influencing training or model selection, except max epochs."""
    return {
        "model": "gat",
        "dataset": "VG150",
        "features": "geometry+trainable_object_embedding",
        "topology": "fully_connected_directed_no_predicate_labels",
        "geometry_dim": args.geometry_dim,
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "num_object_classes": num_objects,
        "num_classes": num_predicates,
        "seed": args.seed,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "negative_ratio": args.negative_ratio,
        "batch_size": args.batch_size,
        "accumulation_steps": args.accumulation_steps,
        "effective_batch_size": args.batch_size * args.accumulation_steps,
        "val_batch_size": args.val_batch_size,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "deterministic": args.deterministic,
        "box_policy": args.box_policy,
        "train_limit": train_limit,
        "val_limit": val_limit,
        "selection_metric": "mR@50",
        "evaluation_protocol": "PredCls_graph_constrained_top1_per_pair",
    }


def main(argv=None):
    args = parse_args(argv)
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    device = resolve_device(args.device)

    with (args.root / "VG-SGG-dicts-with-attri.json").open(encoding="utf-8") as f:
        vocab = json.load(f)
    num_objects = max(vocab["label_to_idx"].values()) + 1
    num_predicates = max(vocab["predicate_to_idx"].values()) + 1
    if num_predicates != 51:
        raise ValueError(f"Expected VG150's 51 classes including background, got {num_predicates}")

    train_dataset = make_dataset(args.root, "train", args.box_policy)
    val_dataset = make_dataset(args.root, "val", args.box_policy)
    train_limit = min(args.max_train_images, len(train_dataset))
    val_limit = min(args.max_val_images, len(val_dataset))
    if train_limit < 1 or val_limit < 1:
        raise ValueError("Train/validation splits must contain images")

    config = frozen_config(args, num_objects, num_predicates, train_limit, val_limit)
    model = build_gat_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(args.seed)
    git_sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()

    start_epoch = 1
    best_score = -math.inf
    best_epoch = 0
    bad_epochs = 0
    history = []
    wandb_id = None
    if args.resume is not None:
        path = args.resume.resolve()
        if path.name != "last.pt":
            raise ValueError("Resume requires last.pt; never best.pt")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("checkpoint_version") != 3 or checkpoint.get("config") != config:
            raise ValueError("GAT checkpoint version or training configuration mismatch")
        run_dir = path.parent
        if not (run_dir / "config.json").is_file():
            raise ValueError("Run directory missing config.json")
        if json.loads((run_dir / "config.json").read_text(encoding="utf-8")) != config:
            raise ValueError("config.json and checkpoint disagree")
        start_epoch, best_score, best_epoch, bad_epochs, history = restore_training_state(
            checkpoint, model, optimizer, generator, device
        )
        if args.epochs < start_epoch:
            raise ValueError("--epochs must be at least the next epoch when resuming")
        wandb_id = checkpoint.get("wandb_id")
        print(f"[RESUME] {path} -> epoch {start_epoch}", flush=True)
    else:
        run_name = args.run_name or f"gat_bs{args.batch_size}_seed{args.seed}"
        run_dir = create_run_dir(args.output_dir, run_name)
        atomic_text_save(json.dumps(config, indent=2) + "\n", run_dir / "config.json")

    print(
        f"[INFO] GAT device={device}; train={train_limit}; val={val_limit}; "
        f"micro_batch={args.batch_size}; effective_batch={args.batch_size * args.accumulation_steps}; run={run_dir}",
        flush=True,
    )
    run = None
    if args.wandb_mode != "disabled":
        import wandb

        if args.resume and checkpoint.get("wandb_id") is None:
            raise ValueError("No previous W&B run to resume; use --wandb-mode disabled")
        run = wandb.init(
            project="semantic-graph-editing",
            name=run_dir.parent.name,
            mode=args.wandb_mode,
            id=wandb_id,
            resume="must" if wandb_id else None,
            config={**config, "git_sha": git_sha, "device": str(device), "max_epochs": args.epochs},
        )
        wandb_id = run.id
    elif args.resume and wandb_id:
        print("[WARN] W&B disabled; previous W&B run not updated", flush=True)

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            train_loss, steps, skipped = train_one_epoch(
                model,
                train_dataset,
                train_limit,
                args.batch_size,
                args.accumulation_steps,
                optimizer,
                args.negative_ratio,
                generator,
                device,
            )
            val = validate_gat(model, val_dataset, val_limit, device, args.val_batch_size)
            score = val["mR@50"]
            if score is None or not math.isfinite(score):
                raise RuntimeError("Cannot select checkpoint without finite validation mR@50")

            stats = {
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
            history.append(stats)
            improved = score > best_score + args.min_delta
            if improved:
                best_score, best_epoch, bad_epochs = score, epoch, 0
            else:
                bad_epochs += 1
            state = {
                "checkpoint_version": 3,
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "sampler_rng": generator.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                "python_rng": random.getstate(),
                "best_score": best_score,
                "best_epoch": best_epoch,
                "bad_epochs": bad_epochs,
                "history": history,
                "config": config,
                "validation": val,
                "git_sha": git_sha,
                "wandb_id": wandb_id,
            }
            if improved:
                atomic_torch_save(state, run_dir / "best.pt")
            atomic_torch_save(state, run_dir / "last.pt")
            atomic_text_save(json.dumps(history, indent=2) + "\n", run_dir / "history.json")
            print(
                f"[Epoch {epoch:03d}/{args.epochs}] loss={train_loss:.4f} "
                f"val_mR50={score:.4f} best={best_score:.4f}@{best_epoch} "
                f"steps={steps} skipped={skipped} {'NEW_BEST' if improved else ''}",
                flush=True,
            )
            if run is not None:
                run.log(stats, step=epoch)
            if bad_epochs >= args.patience:
                print(f"[EARLY STOP] patience={args.patience}", flush=True)
                break
        print(f"[PASS] Best epoch={best_epoch} mR@50={best_score:.4f}", flush=True)
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
