"""Train matched pairwise-KG and logit-adjustment PredCls conditions."""

import argparse
import importlib.metadata
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path

import torch

from scripts.pilot_mlp import atomic_text_save, atomic_torch_save, resolve_device, restore_training_state
from scripts.train_gat import frozen_config, parse_args, train_one_epoch
from scripts.train_mlp import make_dataset
from src.datasets.batching import make_graph_loader
from src.evaluation.gat import validate_gat
from src.knowledge.export import load_class_features
from src.knowledge.transe.core import sha256
from src.models.gat_pair_kg import PairwiseKnowledgeGAT


def main(argv=None):
    # Reuse the original GAT protocol and parse only new experimental controls.
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--kg-features", type=Path, required=False)
    extra.add_argument("--kg-mode", choices=("on", "off"), required=False)
    extra.add_argument("--loss-mode", choices=("ce", "logit_adjusted"), required=False)
    extra.add_argument("--loss-prior", type=Path, required=False)
    extra.add_argument("--logit-tau", type=float, default=0.25)
    tokens = sys.argv[1:] if argv is None else argv
    kg_args, baseline_tokens = extra.parse_known_args(tokens)
    if "--help" in tokens or "-h" in tokens:
        extra.print_help()
    args = parse_args(baseline_tokens)
    if any(value is None for value in (kg_args.kg_features, kg_args.kg_mode, kg_args.loss_mode, kg_args.loss_prior)):
        extra.error("--kg-features, --kg-mode, --loss-mode, and --loss-prior are required")
    if not math.isfinite(kg_args.logit_tau) or kg_args.logit_tau <= 0:
        extra.error("--logit-tau must be positive and finite")
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    device = resolve_device(args.device)
    features, valid, feature_manifest = load_class_features(kg_args.kg_features)
    if feature_manifest["mode"] != "reviewed":
        raise ValueError("Pairwise experiment requires reviewed KG features")
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
    prior = json.loads(kg_args.loss_prior.read_text(encoding="utf-8"))
    if prior.get("schema_version") != 1 or prior.get("split") != "train":
        raise ValueError("Expected version 1 training-only class prior")
    if prior.get("train_limit") != train_limit or prior.get("negative_ratio") != args.negative_ratio:
        raise ValueError("Training prior uses different images or negative sampling ratio")
    hashes = {
        name: sha256(args.root / name)
        for name in ("VG-SGG-with-attri.h5", "VG-SGG-dicts-with-attri.json", "image_data.json", "splits.json")
    }
    if prior.get("data_sha256") != hashes:
        raise ValueError("Training prior data hashes differ from this dataset")
    probabilities = prior.get("class_prior")
    if not isinstance(probabilities, list) or len(probabilities) != num_classes:
        raise ValueError("Training prior has wrong class dimension")
    class_prior = torch.tensor(probabilities, dtype=torch.float32, device=device)
    if (
        not torch.isfinite(class_prior).all()
        or bool((class_prior <= 0).any())
        or abs(class_prior.sum().item() - 1) > 1e-5
    ):
        raise ValueError("Training prior is not a positive probability distribution")
    adjustment = kg_args.logit_tau * class_prior.log() if kg_args.loss_mode == "logit_adjusted" else None
    config = frozen_config(args, num_objects, num_classes, train_limit, val_limit)
    config.update(
        model="gat_pair_kg",
        features=(
            f"geometry+frozen_clip_vit_b32+trainable_object_embedding+pairwise_frozen_{feature_manifest['kg_source']}"
            if args.visual_cache
            else f"geometry+trainable_object_embedding+pairwise_frozen_{feature_manifest['kg_source']}"
        ),
        kg_dim=feature_manifest["dim"],
        kg_manifest_sha256=sha256(kg_args.kg_features / "manifest.json"),
        kg_mode=kg_args.kg_mode,
        loss_mode=kg_args.loss_mode,
        logit_tau=kg_args.logit_tau,
        loss_prior_sha256=sha256(kg_args.loss_prior),
        topology="fully_connected_directed_no_predicate_labels",
        interpretation=feature_manifest["interpretation"],
        max_epochs=args.epochs,
        code_sha256={
            str(path): sha256(path) for root in (Path("src"), Path("scripts")) for path in sorted(root.rglob("*.py"))
        },
        data_sha256=hashes,
    )
    model = PairwiseKnowledgeGAT(config, features, valid).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(args.seed)
    start, best_score, best_epoch, bad_epochs, history = 1, -math.inf, 0, 0, []
    wandb_id = None
    if args.resume:
        path = args.resume.resolve()
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if path.name != "last.pt" or checkpoint.get("checkpoint_version") != 5 or checkpoint.get("config") != config:
            raise ValueError("Resume requires matching pairwise-KG last.pt and exact config/data hashes")
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
        atomic_text_save(json.dumps(prior, indent=2) + "\n", run_dir / "loss_prior.json")
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
    train_loader = (
        make_graph_loader(train, train_limit, args.batch_size, args.num_workers, args.seed + 100_000)
        if args.num_workers
        else None
    )
    val_loader = (
        make_graph_loader(val, val_limit, args.val_batch_size, args.num_workers, args.seed + 200_000)
        if args.num_workers
        else None
    )
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
                train_loader,
                adjustment,
            )
            metrics = validate_gat(model, val, val_limit, device, args.val_batch_size, val_loader)
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
                "checkpoint_version": 5,
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
