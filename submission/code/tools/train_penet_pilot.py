"""Matched CE/PL/PLPR GQA pilot for the frozen-CLIP PE-Net adapter."""

import argparse
import gc
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from src.datasets.gqa_pilot import GQAPilotDataset
from src.knowledge.transe.core import sha256
from src.models.penet import PENetPredicateClassifier
from src.training.gqa_pilot import collate, graph_statistics
from src.training.penet import prototype_losses
from tools.prepare_level2_gqa import write_fixed
from tools.train_gqa_pilot import compact_metrics, evaluate


def state_hash(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(str((tuple(value.shape), str(value.dtype))).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def train_variant(mode, train, val, vectors, config, output, report_dir, device):
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    torch.manual_seed(config["seed"])
    generator = torch.Generator().manual_seed(config["seed"])
    model = PENetPredicateClassifier(*vectors, semantic_dim=config["semantic_dim"], dropout=config["dropout"]).to(
        device
    )
    initial_hash = state_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    run = output / mode
    run.mkdir()
    initial = compact_metrics(evaluate(model, "penet", val, config, device))
    history, best_score, best_epoch = [], -float("inf"), None
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        epoch_start = time.perf_counter()
        order = torch.randperm(len(train), generator=generator).tolist()
        totals = {k: 0.0 for k in ("ce", "pl_euclidean", "pr_cosine", "pr_euclidean", "total")}
        count, steps, skipped, max_norm = 0, 0, 0, 0.0
        trace = hashlib.sha256()
        model.train()
        for begin in range(0, len(order), config["batch_size"]):
            indices = order[begin : begin + config["batch_size"]]
            graphs = [train[i] for i in indices]
            batch, _ = collate(graphs, config["num_classes"], generator)
            trace.update(json.dumps([g.image_id for g in graphs]).encode())
            trace.update(batch["pairs"].numpy().tobytes())
            trace.update(batch["targets"].numpy().tobytes())
            n = len(batch["targets"])
            if n == 0:
                skipped += 1
                continue
            optimizer.zero_grad(set_to_none=True)
            logits, relation, prototypes = model(
                batch["x"].to(device),
                batch["pairs"].to(device),
                batch["object_labels"].to(device),
                return_embeddings=True,
            )
            losses = prototype_losses(logits, relation, prototypes, batch["targets"].to(device), mode=mode)
            if not torch.isfinite(losses["total"]):
                raise RuntimeError("Non-finite loss")
            losses["total"].backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            max_norm = max(max_norm, float(norm))
            optimizer.step()
            for key, value in losses.items():
                totals[key] += float(value.detach()) * n
            count += n
            steps += 1
        if count == 0:
            raise ValueError("No supervised pairs")
        validation = compact_metrics(evaluate(model, "penet", val, config, device))
        score = validation["mR@50"]
        if score is None:
            raise ValueError("Cannot select without validation GT")
        row = {
            "epoch": epoch,
            "train_losses": {k: v / count for k, v in totals.items()},
            "train_pairs": count,
            "steps": steps,
            "skipped_empty_batches": skipped,
            "sampling_trace_sha256": trace.hexdigest(),
            "max_gradient_norm_before_clip": max_norm,
            "validation": validation,
            "wall_seconds": time.perf_counter() - epoch_start,
        }
        history.append(row)
        if score > best_score:
            best_score, best_epoch = score, epoch
            torch.save(
                {"model": model.state_dict(), "config": config, "variant": mode, "epoch": epoch}, run / "best.pt"
            )
        (run / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(
            f"[{mode} {epoch}] total={row['train_losses']['total']:.4f} CE={row['train_losses']['ce']:.4f} "
            f"R50={validation['R@50']:.4f} mR50={score:.5f}",
            flush=True,
        )
    saved = torch.load(run / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(saved["model"])
    final = evaluate(model, "penet", val, config, device, include_ng=True)
    if final["gc"]["mR@50"] != best_score:
        raise AssertionError("Checkpoint reload metric changed")
    report = {
        "status": "PASS",
        "method": "PE-Net adapted to frozen-CLIP PredCls",
        "variant": mode,
        "scope": "one-seed five-epoch engineering pilot, not paper reproduction or main benchmark",
        "config": config,
        "initial_state_sha256": initial_hash,
        "initial_validation": initial,
        "history": history,
        "selected_epoch": best_epoch,
        "validation": final,
        "checkpoint": str(run / "best.pt"),
        "checkpoint_sha256": sha256(run / "best.pt"),
        "parameters": sum(p.numel() for p in model.parameters()),
        "wall_seconds_training_and_final_eval": time.perf_counter() - start,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
        "train_statistics": graph_statistics(train, config["num_classes"]),
        "val_statistics": graph_statistics(val, config["num_classes"]),
        "test_used": False,
        "human_review_samples": 0,
    }
    write_fixed(report_dir / f"{mode}.json", report)
    print(f"[PASS] {mode} selected epoch {best_epoch}", flush=True)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("data/raw/gqa"))
    p.add_argument("--manifest", type=Path, default=Path("data/processed/level2/gqa-v1/manifest.json"))
    p.add_argument("--cohort", type=Path, default=Path("reports/level2/gqa-pilot-v1-cohort.json"))
    p.add_argument("--cache", type=Path, default=Path("data/cache/gqa-pilot-clip-v1"))
    p.add_argument("--glove", type=Path, default=Path("data/processed/level2/gqa-glove-v1"))
    p.add_argument("--output-dir", type=Path, default=Path("checkpoints/level2/penet-pilot-v1"))
    p.add_argument("--report-dir", type=Path, default=Path("reports/level2/penet-pilot-v1"))
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = p.parse_args(argv)
    if args.output_dir.exists() or args.report_dir.exists():
        raise FileExistsError("Use fresh run/report directories")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)
    metadata = json.loads((args.glove / "manifest.json").read_text())
    if metadata["manifest_sha256"] != sha256(args.manifest):
        raise ValueError("GloVe vocabulary manifest mismatch")
    if sha256(args.glove / "vectors.npz") != metadata["arrays_sha256"]:
        raise ValueError("GloVe artifact mismatch")
    with np.load(args.glove / "vectors.npz", allow_pickle=False) as f:
        vectors = [torch.from_numpy(f[k].copy()) for k in ["objects", "predicates"]]
    train = GQAPilotDataset(args.root, args.manifest, args.cohort, args.cache, "train")
    val = GQAPilotDataset(args.root, args.manifest, args.cohort, args.cache, "val")
    if vectors[0].shape != (len(train.vocabulary["label_to_idx"]) + 1, 300) or vectors[1].shape != (
        len(train.vocabulary["predicate_to_idx"]) + 1,
        300,
    ):
        raise ValueError("Invalid mapping dimensions")
    config = {
        "seed": 42,
        "epochs": 5,
        "batch_size": 8,
        "lr": 1e-3,
        "weight_decay": 0.01,
        "dropout": 0.2,
        "feature_dim": 517,
        "semantic_dim": 256,
        "projection_dim": 512,
        "num_classes": len(vectors[1]),
        "num_object_classes": len(vectors[0]),
        "loss_defaults": {"k1": 10, "k2": 1, "gamma1": 1.0, "gamma2": 7.0, "weights": [1, 1, 1, 1]},
        "model": "penet_frozen_clip_pair_proxy_v1",
        "selection_metric": "validation GC mR@50, earliest epoch on ties",
        "sampling": "unchanged GQA pilot multilabel sampling; shared RNG and image order across variants",
        "glove_manifest_sha256": sha256(args.glove / "manifest.json"),
        "glove_vectors_sha256": sha256(args.glove / "vectors.npz"),
        "manifest_sha256": sha256(args.manifest),
        "cohort_sha256": sha256(args.cohort),
        "cache_manifest_sha256": sha256(args.cache / "manifest.json"),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "code_sha256": {
            str(f): sha256(f)
            for f in [
                *sorted(Path("src").rglob("*.py")),
                Path("tools/train_penet_pilot.py"),
                Path("tools/train_gqa_pilot.py"),
            ]
        },
        "torch_version": str(torch.__version__),
        "device": args.device,
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "deterministic_algorithms": True,
        "encoder_frozen": True,
        "upstream_commit": "9c9f50777c66647799cb7a17dfb855aa98aecd1e",
    }
    args.output_dir.mkdir(parents=True)
    args.report_dir.mkdir(parents=True)
    write_fixed(args.report_dir / "config.json", config)
    reports = [
        train_variant(mode, train.graphs, val.graphs, vectors, config, args.output_dir, args.report_dir, args.device)
        for mode in ["ce", "pl", "plpr"]
    ]
    if len({r["initial_state_sha256"] for r in reports}) != 1:
        raise AssertionError("Initial states differ")
    if any(
        [h["sampling_trace_sha256"] for h in r["history"]]
        != [h["sampling_trace_sha256"] for h in reports[0]["history"]]
        for r in reports[1:]
    ):
        raise AssertionError("Sampling differs across variants")
    write_fixed(
        args.report_dir / "matched-controls.json",
        {
            "status": "PASS",
            "identical_initial_state": True,
            "identical_epoch_sampling_traces": True,
            "variants": ["ce", "pl", "plpr"],
            "test_used": False,
        },
    )


if __name__ == "__main__":
    main()
