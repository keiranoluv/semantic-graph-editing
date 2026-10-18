"""Mini-batch VG150 MLP training with CUDA, tqdm and early stopping."""

import argparse
import json
import math
import os
import random
import re
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from scripts.evaluate_mlp import build_model
from scripts.train_mlp import make_dataset
from src.evaluation.batched import evaluate_graph_batch
from src.evaluation.scene_graph import summarize_evaluation
from src.training.sampling import sample_training_pairs


def resolve_device(requested):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(requested)


def collate_training_graphs(graphs, ratio, generator):
    """Sample each image independently, then concatenate selected nodes/pairs."""
    xs = []
    object_labels = []
    pair_indices = []
    targets = []
    node_offset = 0
    skipped = 0

    for graph in graphs:
        selected = sample_training_pairs(
            graph.edge_label,
            negative_ratio=ratio,
            generator=generator,
        )
        if selected.numel() == 0:
            skipped += 1
            continue

        xs.append(graph.x)
        object_labels.append(graph.object_labels)
        pair_indices.append(graph.edge_label_index[:, selected] + node_offset)
        targets.append(graph.edge_label[selected])
        node_offset += graph.num_nodes

    if not targets:
        return None, skipped

    return {
        "x": torch.cat(xs, dim=0),
        "object_labels": torch.cat(object_labels, dim=0),
        "pairs": torch.cat(pair_indices, dim=1),
        "targets": torch.cat(targets, dim=0),
    }, skipped


def forward_batch(model, batch, name, device):
    x = batch["x"].to(device)
    pairs = batch["pairs"].to(device)

    if name == "mlp_semantic":
        labels = batch["object_labels"].to(device)
        return model(x=x, edge_label_index=pairs, object_labels=labels)

    return model(x=x, edge_label_index=pairs)


def train_one_epoch(
    model,
    dataset,
    limit,
    batch_size,
    optimizer,
    ratio,
    generator,
    name,
    device,
):
    model.train()
    total_loss = 0.0
    total_pairs = 0
    steps = 0
    skipped = 0

    progress = tqdm(
        range(0, limit, batch_size),
        desc=f"Train {name}",
        unit="batch",
    )

    for start in progress:
        end = min(start + batch_size, limit)
        graphs = [dataset[i] for i in range(start, end)]

        batch, batch_skipped = collate_training_graphs(graphs, ratio, generator)
        skipped += batch_skipped

        if batch is None:
            continue

        optimizer.zero_grad(set_to_none=True)

        logits = forward_batch(model, batch, name, device)
        targets = batch["targets"].to(device)

        loss = F.cross_entropy(logits, targets)

        if not torch.isfinite(loss).item():
            raise RuntimeError(f"Non-finite loss at batch starting {start}")

        loss.backward()
        optimizer.step()

        count = targets.numel()
        total_loss += loss.detach().item() * count
        total_pairs += count
        steps += 1

        progress.set_postfix(
            loss=f"{total_loss / total_pairs:.4f}",
            images=end,
            steps=steps,
        )

    if steps == 0:
        raise RuntimeError("No successful optimizer steps")

    return total_loss / total_pairs, steps, skipped


@torch.no_grad()
def validate(model, dataset, limit, name, device, batch_size=128):
    model.eval()
    total_loss = 0.0
    total_pairs = 0
    results = []

    progress = tqdm(
        range(0, limit, batch_size),
        desc=f"Val {name}",
        unit="batch",
    )

    for begin in progress:
        end = min(begin + batch_size, limit)
        graphs = [dataset[idx] for idx in range(begin, end)]

        batch_results, loss_sum, pair_count = evaluate_graph_batch(model, graphs, name, device, compute_loss=True)

        results.extend(batch_results)
        total_loss += loss_sum
        total_pairs += pair_count
        progress.set_postfix(images=end)

    if total_pairs == 0:
        raise RuntimeError("No validation candidate pairs")

    metrics = summarize_evaluation(results)
    metrics["loss"] = total_loss / total_pairs
    return metrics


def create_run_dir(output_dir, run_name):
    """Create a new run atomically; never reuse a previous run's outputs."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", run_name):
        raise ValueError("Run name must be a safe directory name")
    root = output_dir / run_name
    root.mkdir(parents=True, exist_ok=True)
    for number in range(1, 100000):
        path = root / f"run_{number:04d}"
        try:
            path.mkdir(exist_ok=False)
            return path
        except FileExistsError:
            continue
    raise RuntimeError("Too many runs with the same name")


def check_resume_compatibility(saved, current):
    """Reject changes to optimization and data protocol on resume."""
    frozen = (
        "model",
        "features",
        "hidden_dim",
        "embedding_dim",
        "num_object_classes",
        "num_classes",
        "box_policy",
        "negative_ratio",
        "batch_size",
        "lr",
        "weight_decay",
        "patience",
        "min_delta",
        "deterministic",
        "seed",
        "train_limit",
        "val_limit",
        "selection_metric",
        "dataset",
    )
    for key in frozen:
        if saved.get(key) != current.get(key):
            raise ValueError(f"Resume config mismatch for {key}: {saved.get(key)!r} != {current.get(key)!r}")


def restore_training_state(checkpoint, model, optimizer, generator, device):
    """Restore all training-critical state at the end of an epoch."""
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    generator.set_state(checkpoint["sampler_rng"])
    torch.set_rng_state(checkpoint["torch_rng"])
    if device.type == "cuda":
        cuda_rng = checkpoint.get("cuda_rng")
        if cuda_rng is None:
            raise ValueError("CUDA RNG state missing from checkpoint")
        torch.cuda.set_rng_state_all(cuda_rng)
    random.setstate(checkpoint["python_rng"])
    return (
        int(checkpoint["epoch"]) + 1,
        float(checkpoint["best_score"]),
        int(checkpoint["best_epoch"]),
        int(checkpoint["bad_epochs"]),
        list(checkpoint["history"]),
    )


def atomic_torch_save(value, path):
    """Replace a completed checkpoint atomically within its own run."""
    temp = path.with_name(path.name + ".tmp")
    try:
        torch.save(value, temp)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def atomic_text_save(content, path):
    temp = path.with_name(path.name + ".tmp")
    try:
        temp.write_text(content, encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["mlp", "mlp_semantic"], required=True)
    parser.add_argument("--root", type=Path, default=Path("data/raw/vg150"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--val-batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--negative-ratio", type=int, default=3)
    parser.add_argument("--max-train-images", type=int, default=57723)
    parser.add_argument("--max-val-images", type=int, default=5000)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--box-policy", choices=["strict", "clip"], default="clip")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/experiments"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--resume", type=Path, default=None, help="Resume a checkpoint created by the new runner (use last.pt)"
    )
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="offline")
    args = parser.parse_args()

    if (
        min(
            args.batch_size,
            args.val_batch_size,
            args.epochs,
            args.patience,
            args.max_train_images,
            args.max_val_images,
            args.hidden_dim,
            args.embedding_dim,
        )
        <= 0
    ):
        parser.error("Batch sizes, epochs, patience, limits and dimensions must be positive")
    if args.lr <= 0 or args.negative_ratio < 0 or args.weight_decay < 0 or args.min_delta < 0:
        parser.error("Invalid learning rate, negative ratio, weight decay or min delta")

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
        vocabulary = json.load(f)
    num_objects = max(vocabulary["label_to_idx"].values()) + 1
    num_classes = max(vocabulary["predicate_to_idx"].values()) + 1
    if num_classes != 51:
        raise ValueError(f"Expected 51 classes; got {num_classes}")

    train_data = make_dataset(args.root, "train", args.box_policy)
    val_data = make_dataset(args.root, "val", args.box_policy)
    train_limit = min(args.max_train_images, len(train_data))
    val_limit = min(args.max_val_images, len(val_data))

    config = {
        "model": args.model,
        "dataset": "VG150",
        "features": "geometry" if args.model == "mlp" else "geometry+trainable_object_embedding",
        "hidden_dim": args.hidden_dim,
        "embedding_dim": args.embedding_dim,
        "num_object_classes": num_objects,
        "num_classes": num_classes,
        "box_policy": args.box_policy,
        "negative_ratio": args.negative_ratio,
        "batch_size": args.batch_size,
        "val_batch_size": args.val_batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "deterministic": args.deterministic,
        "seed": args.seed,
        "train_limit": train_limit,
        "val_limit": val_limit,
        "selection_metric": "mR@50",
    }

    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(args.seed)
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()

    run_name = args.run_name or f"{args.model}_bs{args.batch_size}_seed{args.seed}"
    start_epoch = 1
    best_score = -math.inf
    best_epoch = 0
    bad_epochs = 0
    history = []
    wandb_id = None

    if args.resume:
        checkpoint_path = args.resume.resolve()
        if checkpoint_path.name != "last.pt":
            raise ValueError("Resume requires last.pt, not best.pt")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("checkpoint_version") != 2:
            raise ValueError(
                "Legacy checkpoint detected. Resume is only supported for v2 runs; old files remain untouched"
            )
        check_resume_compatibility(checkpoint["config"], config)
        run_dir = checkpoint_path.parent
        if not (run_dir / "config.json").is_file():
            raise ValueError("Run directory missing config.json")
        if json.loads((run_dir / "config.json").read_text()) != checkpoint["config"]:
            raise ValueError("Run config.json differs from checkpoint")
        start_epoch, best_score, best_epoch, bad_epochs, history = restore_training_state(
            checkpoint, model, optimizer, generator, device
        )
        if args.epochs < start_epoch:
            raise ValueError("--epochs must be >= the next epoch when resuming")
        wandb_id = checkpoint.get("wandb_id")
        print(f"[RESUME] {checkpoint_path} -> epoch {start_epoch}", flush=True)
    else:
        run_dir = create_run_dir(args.output_dir, run_name)
        (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    print(
        f"[INFO] model={args.model} device={device} batch_size={args.batch_size} val_batch_size={args.val_batch_size}",
        flush=True,
    )
    print(f"[INFO] train={train_limit} val={val_limit} output={run_dir}", flush=True)

    run = None
    if args.wandb_mode != "disabled":
        import wandb

        if args.resume and not wandb_id:
            raise ValueError("Cannot resume W&B logging: previous run has no W&B ID; use --wandb-mode disabled")
        run = wandb.init(
            project="semantic-graph-editing",
            name=run_name,
            mode=args.wandb_mode,
            id=wandb_id,
            resume="must" if wandb_id else None,
            config={**config, "git_sha": sha, "device": str(device)},
        )
        wandb_id = run.id
    elif args.resume and wandb_id:
        print("[WARN] W&B is disabled; original run will not be updated", flush=True)

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            train_loss, steps, skipped = train_one_epoch(
                model,
                train_data,
                train_limit,
                args.batch_size,
                optimizer,
                args.negative_ratio,
                generator,
                args.model,
                device,
            )
            val = validate(model, val_data, val_limit, args.model, device, args.val_batch_size)
            score = val["mR@50"]
            if score is None or not math.isfinite(score):
                raise RuntimeError("Invalid mR@50: cannot select checkpoint")

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

            improved = score > best_score + args.min_delta
            if improved:
                best_score = score
                best_epoch = epoch
                bad_epochs = 0
            else:
                bad_epochs += 1

            checkpoint = {
                "checkpoint_version": 2,
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
                "git_sha": sha,
                "wandb_id": wandb_id,
            }

            if improved:
                atomic_torch_save(checkpoint, run_dir / "best.pt")
            atomic_torch_save(checkpoint, run_dir / "last.pt")
            atomic_text_save(json.dumps(history, indent=2) + "\n", run_dir / "history.json")

            print(
                f"[Epoch {epoch:03d}/{args.epochs}] "
                f"train={train_loss:.4f} val_loss={val['loss']:.4f} "
                f"R50={val['R@50']:.4f} mR50={score:.4f} "
                f"best={best_score:.4f}@{best_epoch} steps={steps} "
                f"{'NEW_BEST' if improved else ''}",
                flush=True,
            )
            if run is not None:
                run.log(metrics, step=epoch)
            if bad_epochs >= args.patience:
                print(f"[EARLY STOP] patience={args.patience}", flush=True)
                break

        print(f"[PASS] Best epoch={best_epoch} mR@50={best_score:.4f}", flush=True)
        print(f"[INFO] Checkpoints: {run_dir}", flush=True)
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
