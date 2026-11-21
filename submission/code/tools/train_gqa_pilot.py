"""Matched frozen-CLIP GQA MLP/GAT pilot; never load or tune on project test."""

import argparse
import json
import os
import random
import subprocess
import time
from pathlib import Path

import torch

from src.datasets.gqa_pilot import GQAPilotDataset
from src.evaluation.gqa_pilot import evaluate_gqa_image, summarize_gqa
from src.knowledge.transe.core import sha256
from src.models.gat import GATPredicateClassifier
from src.models.mlp_semantic import SemanticMLPPredicateClassifier
from src.training.gqa_pilot import collate, cross_entropy_sum, forward, graph_statistics
from tools.prepare_level2_gqa import write_fixed


def build_model(kind, config):
    kwargs = {
        k: config[k] for k in ("geometry_dim", "num_object_classes", "embedding_dim", "hidden_dim", "num_classes")
    }
    if kind == "gat":
        return GATPredicateClassifier(**kwargs, num_heads=4, num_layers=2, dropout=0.1)
    if kind == "mlp":
        return SemanticMLPPredicateClassifier(**kwargs)
    raise ValueError("Unknown model")


@torch.inference_mode()
def evaluate(model, kind, graphs, config, device, *, include_ng=False):
    model.eval()
    gc, ng = [], []
    loss_sum, count = 0.0, 0
    for begin in range(0, len(graphs), config["batch_size"]):
        group = graphs[begin : begin + config["batch_size"]]
        batch, slices = collate(group, config["num_classes"])
        logits = forward(model, kind, batch, device).cpu()
        loss_sum += float(cross_entropy_sum(logits, batch["targets"]))
        count += len(logits)
        for graph, (start, end) in zip(group, slices, strict=True):
            gc.append(evaluate_gqa_image(logits[start:end], graph, config["num_classes"] - 1))
            if include_ng:
                ng.append(
                    evaluate_gqa_image(logits[start:end], graph, config["num_classes"] - 1, graph_constraint=False)
                )
    if not count:
        raise ValueError("No validation candidate pairs")
    result = {"loss_all_pairs": loss_sum / count, "gc": summarize_gqa(gc, config["num_classes"] - 1)}
    if include_ng:
        result["ng"] = summarize_gqa(ng, config["num_classes"] - 1, graph_constraint=False)
    return result


def compact_metrics(result):
    return {
        "loss_all_pairs": result["loss_all_pairs"],
        **{k: result["gc"][k] for k in ("R@50", "R@100", "mR@50", "mR@100")},
    }


def train_model(kind, train, val, config, run_dir, report_dir, device):
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])
    generator = torch.Generator().manual_seed(config["seed"])
    model = build_model(kind, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    model_dir = run_dir / kind
    model_dir.mkdir()
    start = time.perf_counter()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    initial = compact_metrics(evaluate(model, kind, val, config, device))
    history, best_score, best_epoch = [], -float("inf"), None
    for epoch in range(1, config["epochs"] + 1):
        epoch_start = time.perf_counter()
        order = torch.randperm(len(train), generator=generator).tolist()
        model.train()
        loss_sum, count, steps, skipped_batches = 0.0, 0, 0, 0
        for begin in range(0, len(order), config["batch_size"]):
            batch, _ = collate(
                [train[i] for i in order[begin : begin + config["batch_size"]]], config["num_classes"], generator
            )
            n = len(batch["targets"])
            if not n:
                skipped_batches += 1
                continue
            optimizer.zero_grad(set_to_none=True)
            logits = forward(model, kind, batch, device)
            loss = cross_entropy_sum(logits, batch["targets"].to(device)) / n
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, error_if_nonfinite=True)
            optimizer.step()
            loss_sum += float(loss.detach()) * n
            count += n
            steps += 1
        if not count:
            raise ValueError("No training candidates")
        validation = compact_metrics(evaluate(model, kind, val, config, device))
        score = validation["mR@50"]
        if score is None:
            raise ValueError("Validation has no GT; cannot select checkpoint")
        row = {
            "epoch": epoch,
            "train_loss_sampled_pairs": loss_sum / count,
            "train_pairs": count,
            "optimizer_steps": steps,
            "skipped_empty_batches": skipped_batches,
            "last_gradient_norm_before_clip": float(grad_norm),
            "validation": validation,
            "wall_seconds": time.perf_counter() - epoch_start,
        }
        history.append(row)
        if score > best_score:
            best_score, best_epoch = score, epoch
            torch.save(
                {"model": model.state_dict(), "kind": kind, "config": config, "epoch": epoch}, model_dir / "best.pt"
            )
        write_path = model_dir / "history.json"
        write_path.write_text(json.dumps(history, indent=2) + "\n")
        print(
            f"[{kind} {epoch}/{config['epochs']}] loss={loss_sum/count:.4f} "
            f"val_R50={validation['R@50']:.4f} val_mR50={score:.4f} seconds={row['wall_seconds']:.1f}",
            flush=True,
        )
    checkpoint = torch.load(model_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    final = evaluate(model, kind, val, config, device, include_ng=True)
    if final["gc"]["mR@50"] != best_score:
        raise AssertionError("Reloaded best checkpoint evaluation differs")
    report = {
        "status": "PASS",
        "scope": "small one-seed engineering pilot, not final benchmark",
        "model": kind,
        "config": config,
        "initial_validation": initial,
        "history": history,
        "selected_epoch": best_epoch,
        "validation": final,
        "checkpoint": str(model_dir / "best.pt"),
        "checkpoint_sha256": sha256(model_dir / "best.pt"),
        "parameters": sum(p.numel() for p in model.parameters()),
        "wall_seconds": time.perf_counter() - start,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
        "train_statistics": graph_statistics(train, config["num_classes"]),
        "val_statistics": graph_statistics(val, config["num_classes"]),
        "test_used": False,
        "human_review_samples": 0,
    }
    write_fixed(report_dir / f"{kind}.json", report)
    print(f"[PASS] {kind}: best epoch {best_epoch}; report={report_dir / (kind + '.json')}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/gqa"))
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/level2/gqa-v1/manifest.json"))
    parser.add_argument("--cohort", type=Path, default=Path("reports/level2/gqa-pilot-v1-cohort.json"))
    parser.add_argument("--cache", type=Path, default=Path("data/cache/gqa-pilot-clip-v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/level2/gqa-pilot-v1"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/level2/gqa-pilot-v1"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size) <= 0:
        parser.error("epochs/batch-size must be positive")
    if args.output_dir.exists() or args.report_dir.exists():
        raise FileExistsError("Use fresh run/report directories; existing pilot results are immutable")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(1)
    train = GQAPilotDataset(args.root, args.manifest, args.cohort, args.cache, "train")
    val = GQAPilotDataset(args.root, args.manifest, args.cohort, args.cache, "val")
    vocab = train.vocabulary
    code_files = sorted(
        set(Path("src").rglob("*.py"))
        | {
            Path("tools/train_gqa_pilot.py"),
            Path("tools/cache_gqa_pilot.py"),
            Path("tools/prepare_level2_gqa.py"),
            Path("scripts/cache_vg150_regions.py"),
        }
    )
    config = {
        "schema_version": 1,
        "dataset": "GQA_custom_full_train_vocab_pilot_v1",
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": 1e-3,
        "weight_decay": 0.01,
        "geometry_dim": 517,
        "embedding_dim": 32,
        "hidden_dim": 256,
        "num_object_classes": max(vocab["label_to_idx"].values()) + 1,
        "num_classes": max(vocab["predicate_to_idx"].values()) + 1,
        "supervision": "uniform distinct GT predicates per pair; soft-target CE; duplicates carry no extra train weight",
        "sampling": "all positive pairs + up to 3x backgrounds; no-GT images up to 32 backgrounds; zero-pair images no loss",
        "topology": "all directed pairs without self edges; GAT adds self loops internally; never GT relations",
        "selection_metric": "validation GC mR@50 over all pinned predicates, earliest epoch on tie",
        "gradient_clip_norm": 5.0,
        "deterministic_algorithms": True,
        "manifest_sha256": sha256(args.manifest),
        "cohort_sha256": sha256(args.cohort),
        "cache_manifest_sha256": sha256(args.cache / "manifest.json"),
        "code_sha256": {str(f): sha256(f) for f in code_files},
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "torch_version": str(torch.__version__),
        "device": args.device,
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "gat": {"num_heads": 4, "num_layers": 2, "dropout": 0.1},
    }
    args.output_dir.mkdir(parents=True)
    args.report_dir.mkdir(parents=True)
    write_fixed(args.report_dir / "config.json", config)
    write_fixed(args.report_dir / "cache.json", json.loads((args.cache / "manifest.json").read_text()))
    print(f"[READY] train={len(train)} val={len(val)} classes={config['num_classes']}", flush=True)
    for kind in ("mlp", "gat"):
        train_model(kind, train.graphs, val.graphs, config, args.output_dir, args.report_dir, args.device)


if __name__ == "__main__":
    main()
