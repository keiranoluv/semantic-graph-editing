"""Train GAT+WordNet with traceable inputs and matched ablation switches."""

import argparse
import importlib.metadata
import json
import math
import random
import subprocess
import sys
from pathlib import Path

import torch

from scripts.pilot_mlp import atomic_text_save, atomic_torch_save, resolve_device, restore_training_state
from scripts.train_gat import frozen_config, parse_args, train_one_epoch
from scripts.train_mlp import make_dataset
from src.evaluation.gat import validate_gat
from src.knowledge.export import load_class_features
from src.knowledge.transe.core import sha256
from src.models.gat_kg import KnowledgeGAT


def main(argv=None):
    # Reuse baseline arguments/validation without editing its source or runner.
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--kg-features", type=Path)
    extra.add_argument("--allow-draft", action="store_true")
    extra.add_argument("--no-knowledge", action="store_true")
    extra.add_argument("--message-edges", choices=("full", "none"), default="full")
    tokens = sys.argv[1:] if argv is None else argv
    kg_args, baseline_tokens = extra.parse_known_args(tokens)
    if "--help" in tokens or "-h" in tokens:
        extra.print_help()
    args = parse_args(baseline_tokens)
    if kg_args.kg_features is None:
        extra.error("--kg-features is required")
    if args.deterministic:
        import os

        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    device = resolve_device(args.device)
    features, valid, feature_manifest = load_class_features(kg_args.kg_features, allow_draft=kg_args.allow_draft)
    vocab_path = args.root / "VG-SGG-dicts-with-attri.json"
    if sha256(vocab_path) != feature_manifest["input_sha256"]["vocabulary"]:
        raise ValueError("Dataset vocabulary differs from exported feature vocabulary")
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    num_objects = max(vocab["label_to_idx"].values()) + 1
    num_classes = max(vocab["predicate_to_idx"].values()) + 1
    if num_objects != 151 or num_classes != 51:
        raise ValueError("Expected VG150 vocabulary (151 objects / 51 predicates with background)")
    train = make_dataset(args.root, "train", args.box_policy, args.visual_cache)
    val = make_dataset(args.root, "val", args.box_policy, args.visual_cache)
    train_limit, val_limit = min(args.max_train_images, len(train)), min(args.max_val_images, len(val))
    config = frozen_config(args, num_objects, num_classes, train_limit, val_limit)
    config.update(
        model="gat_kg",
        features=(
            "geometry+frozen_clip_vit_b32+trainable_object_embedding+frozen_wordnet+valid_bit"
            if args.visual_cache
            else "geometry+trainable_object_embedding+frozen_wordnet+valid_bit"
        ),
        kg_dim=feature_manifest["dim"],
        kg_manifest_sha256=sha256(kg_args.kg_features / "manifest.json"),
        use_knowledge=not kg_args.no_knowledge,
        message_edges=kg_args.message_edges,
        topology="fully_connected_directed" if kg_args.message_edges == "full" else "self_loops_only",
        interpretation=feature_manifest["interpretation"],
        max_epochs=args.epochs,
        code_sha256={
            str(path): sha256(path) for root in (Path("src"), Path("scripts")) for path in sorted(root.rglob("*.py"))
        },
        data_sha256={
            name: sha256(args.root / name)
            for name in ["VG-SGG-with-attri.h5", "VG-SGG-dicts-with-attri.json", "image_data.json", "splits.json"]
        },
    )
    model = KnowledgeGAT(config, features, valid).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(args.seed)
    start, best_score, best_epoch, bad_epochs, history = 1, -math.inf, 0, 0, []
    wandb_id = None
    if args.resume:
        path = args.resume.resolve()
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if path.name != "last.pt" or checkpoint.get("checkpoint_version") != 4 or checkpoint.get("config") != config:
            raise ValueError("Resume requires matching GAT+KG last.pt and exact config/data hashes")
        run_dir = path.parent
        if json.loads((run_dir / "config.json").read_text()) != config:
            raise ValueError("Resume config.json mismatch")
        start, best_score, best_epoch, bad_epochs, history = restore_training_state(
            checkpoint, model, optimizer, generator, device
        )
        wandb_id = checkpoint.get("wandb_id")
        if start > args.epochs or bad_epochs >= args.patience:
            raise ValueError("Run has already completed its epoch budget or early stopping")
    else:
        if args.run_name is not None:
            extra.error("Use --output-dir as the exclusive run directory; --run-name is unsupported")
        run_dir = args.output_dir
        run_dir.mkdir(parents=True, exist_ok=False)
        atomic_text_save(json.dumps(config, indent=2) + "\n", run_dir / "config.json")
        atomic_text_save(json.dumps(feature_manifest, indent=2) + "\n", run_dir / "kg_manifest.json")
        versions = {name: importlib.metadata.version(name) for name in ["torch", "torch-geometric", "numpy", "h5py"]}
        atomic_text_save(
            json.dumps({"python": sys.version, "packages": versions}, indent=2), run_dir / "environment.json"
        )
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
        atomic_text_save(freeze.stdout, run_dir / "requirements-frozen.txt")
    git_sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    print(
        f"[{feature_manifest['mode'].upper()}] device={device}; train={train_limit}; val={val_limit}; run={run_dir}",
        flush=True,
    )
    run = None
    if args.wandb_mode != "disabled":
        import wandb

        if args.resume and wandb_id is None:
            raise ValueError("No previous W&B run to resume; use --wandb-mode disabled")
        run = wandb.init(
            project="semantic-graph-editing",
            name=run_dir.name,
            mode=args.wandb_mode,
            id=wandb_id,
            resume="must" if wandb_id else None,
            config={**config, "git_sha": git_sha, "device": str(device)},
        )
        wandb_id = run.id
    elif args.resume and wandb_id:
        print("[WARN] W&B disabled; previous W&B run not updated", flush=True)
    try:
        for epoch in range(start, args.epochs + 1):
            loss, steps, skipped = train_one_epoch(
                model,
                train,
                train_limit,
                args.batch_size,
                args.accumulation_steps,
                optimizer,
                args.negative_ratio,
                generator,
                device,
            )
            metrics = validate_gat(model, val, val_limit, device, args.val_batch_size)
            score = metrics["mR@50"]
            if score is None or not math.isfinite(score):
                raise RuntimeError("Cannot select checkpoint without finite validation mR@50")
            stats = {
                "epoch": epoch,
                "train/loss": loss,
                "train/optimizer_steps": steps,
                "train/skipped_graphs": skipped,
                **{f"val/{k}": v for k, v in metrics.items()},
            }
            history.append(stats)
            improved = score > best_score + args.min_delta
            if improved:
                best_score, best_epoch, bad_epochs = score, epoch, 0
            else:
                bad_epochs += 1
            state = {
                "checkpoint_version": 4,
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "sampler_rng": generator.get_state(),
                "torch_rng": torch.get_rng_state(),
                "python_rng": random.getstate(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "bad_epochs": bad_epochs,
                "history": history,
                "config": config,
                "validation": metrics,
                "git_sha": git_sha,
                "wandb_id": wandb_id,
            }
            if improved:
                atomic_torch_save(state, run_dir / "best.pt")
            atomic_torch_save(state, run_dir / "last.pt")
            atomic_text_save(json.dumps(history, indent=2) + "\n", run_dir / "history.json")
            print(
                f"[Epoch {epoch}] loss={loss:.5f} val_mR50={score:.5f} best={best_score:.5f}@{best_epoch}", flush=True
            )
            if run is not None:
                run.log(stats, step=epoch)
            if bad_epochs >= args.patience:
                print(f"[EARLY STOP] patience={args.patience}", flush=True)
                break
        print(f"[PASS] Best epoch={best_epoch} mR@50={best_score:.5f}", flush=True)
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
