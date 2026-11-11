"""Shared frozen-checkpoint evaluation for MLP, GAT and GAT+KG (val by default)."""

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
import numpy as np

from scripts.evaluate_mlp import build_model
from scripts.pilot_mlp import resolve_device
from scripts.train_mlp import make_dataset
from src.datasets.batching import graph_batches, make_graph_loader
from src.evaluation.batched import evaluate_graph_batch
from src.evaluation.gat import evaluate_gat_batch
from src.evaluation.scene_graph import summarize_evaluation
from src.evaluation.reference_predcls import NG_PROTOCOL, PROTOCOL, REFERENCE_COMMIT, summarize_reference
from src.features.visual import load_visual_manifest
from src.knowledge.transe.core import sha256
from src.models.gat import build_gat_model
from src.models.gat_kg import model_from_checkpoint
from src.models.gat_kg_fusion import model_from_checkpoint as fusion_model_from_checkpoint
from src.models.gat_pair_kg import model_from_checkpoint as pair_model_from_checkpoint


def load_model(checkpoint):
    """Reconstruct every level 1 baseline and the new KG model."""
    config = checkpoint["config"]
    name = config["model"]
    if name == "gat_kg":
        return model_from_checkpoint(checkpoint)
    if name == "gat_pair_kg":
        return pair_model_from_checkpoint(checkpoint)
    if name == "gat_kg_fusion":
        return fusion_model_from_checkpoint(checkpoint)
    if name == "gat":
        model = build_gat_model(config)
    else:
        model = build_model(config)
    model.load_state_dict(checkpoint["model"])
    return model


@torch.inference_mode()
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--root", type=Path, default=Path("data/raw/vg150"))
    p.add_argument("--visual-cache", type=Path, help="Override cache path stored in a visual checkpoint")
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--allow-test", action="store_true")
    p.add_argument(
        "--protocol",
        choices=("legacy", "reference-v1", "reference-ng-v1"),
        default="legacy",
        help="reference-v1: one predicate per pair; reference-ng-v1: all predicates; legacy: historical scoring",
    )
    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0, help="Persistent graph loading workers")
    p.add_argument("--max-images", type=int)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    if args.split == "test" and not args.allow_test:
        p.error("Held-out test requires --allow-test after experimental setup is frozen")
    if args.batch_size <= 0 or args.num_workers < 0 or (args.max_images is not None and args.max_images <= 0):
        p.error("Batch size and optional image limit must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    config = checkpoint["config"]
    name = config["model"]
    if args.protocol != "legacy" and config["num_classes"] != 51:
        raise ValueError("reference-v1 requires the fixed VG150 vocabulary of 50 predicates")
    data_hashes = {
        name: sha256(args.root / name)
        for name in ["VG-SGG-with-attri.h5", "VG-SGG-dicts-with-attri.json", "image_data.json", "splits.json"]
    }
    if config.get("data_sha256") is not None and config["data_sha256"] != data_hashes:
        raise ValueError("Evaluation data differs from training input hashes")
    visual_cache = args.visual_cache or config.get("visual_cache")
    if config.get("visual_cache_manifest_sha256"):
        if visual_cache is None:
            raise ValueError("Visual checkpoint requires its frozen region cache")
        visual_cache = Path(visual_cache)
        load_visual_manifest(visual_cache, args.root, config["box_policy"])
        if sha256(visual_cache / "manifest.json") != config["visual_cache_manifest_sha256"]:
            raise ValueError("Visual cache differs from training checkpoint")
    elif args.visual_cache is not None:
        raise ValueError("Cannot add visual features to an annotation-only checkpoint")
    model = load_model(checkpoint)
    model.to(device).eval()
    dataset = make_dataset(args.root, args.split, config["box_policy"], visual_cache)
    limit = len(dataset) if args.max_images is None else min(args.max_images, len(dataset))
    loader = (
        make_graph_loader(dataset, limit, args.batch_size, args.num_workers, config["seed"] + 300_000)
        if args.num_workers
        else None
    )
    results = []
    for _, graphs in graph_batches(dataset, limit, args.batch_size, loader):
        if name in {"gat", "gat_kg", "gat_pair_kg", "gat_kg_fusion"}:
            batch_results, _, _ = evaluate_gat_batch(model, graphs, device, protocol=args.protocol)
        else:
            batch_results, _, _ = evaluate_graph_batch(model, graphs, name, device, protocol=args.protocol)
        results.extend(batch_results)
    support = Counter()
    hits = {50: Counter(), 100: Counter()}
    for result in results:
        support.update(result["class_support"])
        for k in hits:
            hits[k].update(result["class_hits"][k])
    summary = (
        summarize_evaluation(results)
        if args.protocol == "legacy"
        else summarize_reference(
            results, num_predicates=config["num_classes"] - 1, graph_constraint=args.protocol != "reference-ng-v1"
        )
    )
    report = {
        **summary,
        "model": name,
        "seed": config["seed"],
        "split": args.split,
        "images_evaluated": limit,
        "split_images": len(dataset),
        "full_split": limit == len(dataset),
        "evaluation_device": str(device),
        "evaluation_batch_size": args.batch_size,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint": str(args.checkpoint),
        "config": config,
        "data_sha256": data_hashes,
        "historical_training_data_hashes_available": "data_sha256" in config,
        "protocol": {
            "legacy": "PredCls_graph_constrained_top1_per_pair",
            "reference-v1": PROTOCOL,
            "reference-ng-v1": NG_PROTOCOL,
        }[args.protocol],
        "recall_aggregation": "image_macro",
        "mean_recall_aggregation": "predicate_macro_over_gt_support"
        if args.protocol == "legacy"
        else "image_macro_per_predicate_then_fixed_vocabulary_macro",
        "interpretation": config.get("interpretation", "baseline_evaluation"),
        "per_predicate": {
            str(c): {"support": support[c], **{f"R@{k}": hits[k][c] / support[c] for k in hits}}
            for c in sorted(support)
        },
    }
    if args.protocol != "legacy":
        report["per_predicate"] = summary["per_predicate"]
        report["evaluator_version"] = args.protocol
        report["graph_constraint"] = args.protocol == "reference-v1"
        report["reference_commit"] = REFERENCE_COMMIT
        report["matching"] = "GT_object_class_and_subject_object_IoU_ge_0.5_inclusive_xyxy"
        report["gt_duplicate_policy"] = "retain_annotation_rows"
        report["ranking"] = "foreground_max_full_softmax_unit_object_scores_torch_sort_descending"
        if args.protocol == "reference-ng-v1":
            report["ranking"] = "all_foreground_full_softmax_unit_object_scores_numpy_argsort_after_pair_torch_sort"
        report["checkpoint_selection_protocol"] = config.get("evaluation_protocol", "historical_unspecified")
        report["interpretation"] = "fixed_checkpoint_rescore_not_reselected_under_reference_protocol"
        report["evaluator_sha256"] = {
            file: sha256(Path(file))
            for file in (
                "src/evaluation/reference_predcls.py",
                "src/evaluation/protocols.py",
                "scripts/evaluate_level1.py",
                "src/evaluation/gat.py",
                "src/evaluation/batched.py",
            )
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    prefix = "ng-" if args.protocol == "reference-ng-v1" else ""
    print(
        json.dumps(
            {k: report[k] for k in ["model", "seed", "images_evaluated", f"{prefix}R@50", f"{prefix}mR@50"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
