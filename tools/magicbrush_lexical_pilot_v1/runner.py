"""Frozen-feature lexical pilot, explicitly authorized for assistant GPU execution."""
import argparse
import gc
import json
from collections import Counter
from pathlib import Path

import torch

from src.datasets.gqa_full import atomic_write
from tools.edit_model_v1.text import load_store
from tools.emnlp2020_graph_modification_v1.dataset import read, sha256, write
from tools.magicbrush_compact_pilot_v1.data import evaluation_report, partition
from tools.magicbrush_compact_pilot_v1.runner import preflight as base_preflight
from tools.magicbrush_compact_pilot_v1.runner import source_hashes as base_hashes
from tools.magicbrush_experiment_v1.evaluation import score_rows
from tools.magicbrush_experiment_v3.pipeline import versions
from tools.magicbrush_experiment_v3.vocabulary import prototype_vectors
from tools.prepare_magicbrush import ROOT, digest
from tools.run_gqa_full import exclusive

from .analysis import diagnose
from .features import EDGE_FEATURES, LABEL_FEATURES, NODE_FEATURES, GroundingStore, collate_inputs
from .model import StructuredPredictor
from .runtime import validate
from .training import train_one

CONFIG = ROOT / "configs/level2/magicbrush-lexical-pilot-v1.json"
FILES = (
    "best.pt",
    "last.pt",
    "result.json",
    "predictions-val.json",
    "predictions-val-all.json",
    "validation-report.json",
    "ablation.json",
)


def source_hashes():
    paths = [*Path(__file__).parent.glob("*.py"), ROOT / "scripts/run_magicbrush_lexical_pilot_v1.sh"]
    return {**base_hashes(), **{str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}}


def preflight():
    cfg = read(CONFIG)
    if read(CONFIG.with_suffix(".lock.json")) != {
        "config_sha256": sha256(CONFIG),
        "source_sha256": source_hashes(),
        "versions": versions(),
    }:
        raise ValueError("Lexical release changed")
    parent, examples, validation, parent_identity, parent_report = base_preflight()
    for k, v in parent.items():
        if k not in ("schema", "output") and cfg[k] != v:
            raise ValueError("Changed baseline recipe: " + k)
    if cfg["schema"] != "magicbrush-lexical-pilot-v1" or cfg["output"] != "runs/level2/magicbrush-lexical-pilot-v1":
        raise ValueError("Expected isolated lexical output")
    for name, spec in cfg["provenance"].items():
        if sha256(ROOT / spec["path"]) != spec["sha256"]:
            raise ValueError("Provenance changed: " + name)
    store, meta = load_store(ROOT / parent["text_cache"], parent_identity)
    if meta["source_input_ids"] != [e.id for e in examples["train"] + validation]:
        raise ValueError("Frozen feature input IDs changed")
    wrapped = GroundingStore(store, parent_identity["vocabulary"])
    identity = {
        **parent_identity,
        "schema": cfg["schema"],
        "config": cfg,
        "release_lock_sha256": sha256(CONFIG.with_suffix(".lock.json")),
        "baseline_identity_sha256": digest(parent_identity),
        "frozen_cache_manifest_sha256": sha256(ROOT / parent["text_cache"] / "manifest.json"),
        "frozen_vectors_sha256": meta["vectors_sha256"],
        "grounding": "input-only soft trainable lexical label/action residuals; fixed train vocabulary; no hard label/action filtering",
    }
    # Only train target labels are used for this feature-availability diagnostic.
    counts = Counter()
    labels = identity["vocabulary"]["labels"]
    for start in range(0, len(examples["train"]), 32):
        batch = examples["train"][start : start + 32]
        features = collate_inputs([e.model_input() for e in batch], wrapped)
        for b, e in enumerate(batch):
            for op in e.operations:
                if op.kind == "add_node":
                    counts["train_added_nodes"] += 1
                    counts["train_added_name_exact_mention"] += bool(
                        features["label_grounding"][b, labels.index(op.after.label), 0]
                    )
                elif op.kind in ("delete_node", "replace_node"):
                    i = next(i for i, n in enumerate(e.source.nodes) if n.id == op.entity_id)
                    counts["train_changed_source_nodes"] += 1
                    counts["train_changed_source_name_exact_mention"] += bool(features["node_grounding"][b, i, 0])
    report = {
        **parent_report,
        "experiment_sha256": digest(identity),
        "queue": [
            "MLP/42 training + validation + inference ablations",
            "GNN/42 training + validation + inference ablations",
        ],
        "cache_reused_without_reencoding": True,
        "cache_texts": len(meta["text_sha256"]),
        "train_grounding_coverage": dict(counts),
        "feature_names": {"labels": LABEL_FEATURES, "nodes": NODE_FEATURES, "edges": EDGE_FEATURES},
        "mechanism": "Soft additive logit bonuses with trainable weights initialized to declared lexical priors. Unmentioned labels remain possible; OOV names cannot be emitted.",
        "ablation_scope": "Disable label residuals, ID residuals or both at selected checkpoint; inference-only diagnostic, not separately retrained ablations",
        "authorization": "User explicitly allowed assistant to run and analyze this pilot; CPU checks and code commit precede GPU",
        "test_labels_decoded": False,
        "gpu_jobs_started": False,
    }
    return cfg, examples, validation, wrapped, identity, report


def completed(directory, run_id):
    receipt = read(directory / "completed-receipt.json")
    if receipt["identity"] != run_id or set(receipt["files"]) != set(FILES):
        raise ValueError("Completion identity/files differ")
    for name, h in receipt["files"].items():
        if sha256(directory / name) != h:
            raise ValueError("Completed artifact modified")
    return (
        read(directory / "result.json"),
        read(directory / "validation-report.json"),
        read(directory / "ablation.json"),
    )


def finish(directory, result, validation, store, cfg, identity, kind, *, device="cuda"):
    model = StructuredPredictor(
        kind=kind,
        vocabulary=identity["vocabulary"],
        prototypes=prototype_vectors(store, identity["vocabulary"]),
        text_dim=store.dim,
        **cfg["model"],
    ).to(device)
    saved = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(saved["model"])
    rows = read(directory / "predictions-val.json")
    scored, gaps = partition(validation)
    if gaps:
        _, extra = validate(model, gaps, store, cfg, device)
        rows += extra
    by_id = {r["id"]: r for r in rows}
    all_rows = [by_id[e.id] for e in validation]
    measured = evaluation_report(validation, all_rows)
    if any(measured["constrained"][k] != v for k, v in result["validation"].items()):
        raise ValueError("Selected metrics differ")
    if any(measured["raw"][k] != v for k, v in result["raw_validation"].items()):
        raise ValueError("Selected raw metrics differ")
    measured["grounding_diagnostics"] = diagnose(scored, [by_id[e.id] for e in scored])
    ablations = {}
    for name, labels, ids in [
        ("without_labels", False, True),
        ("without_ids", True, False),
        ("without_both", False, False),
    ]:
        model.enable_labels = labels
        model.enable_ids = ids
        _, pred = validate(model, scored, store, cfg, device)
        ablations[name] = {"constrained": score_rows(scored, pred), "diagnostics": diagnose(scored, pred)}
    ablation = {
        "scope": "inference-only ablation at same selected weights; not retrained models",
        "results": ablations,
        "learned_weights": {
            k: v.detach().cpu().tolist()
            for k, v in model.named_parameters()
            if k in ("new_label_weights", "replace_label_weights", "node_action_weights", "edge_action_weights")
        },
    }
    write(directory / "predictions-val-all.json", all_rows)
    write(directory / "validation-report.json", measured)
    write(directory / "ablation.json", ablation)
    write(
        directory / "completed-receipt.json",
        {"identity": result["identity"], "files": {n: sha256(directory / n) for n in FILES}},
    )
    return measured, ablation


def execute(cfg, examples, validation, store, identity, report):
    output = ROOT / cfg["output"]
    with exclusive(ROOT / "runs/level2/.gqa-kg-training-device"), exclusive(output):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        torch.set_num_threads(1)
        torch.backends.cudnn.benchmark = False
        write(output / "experiment.json", identity)
        write(output / "preflight.json", report)
        results = {}
        try:
            for kind in cfg["models"]:
                directory = output / kind / str(cfg["seed"])
                run_id = {"experiment_sha256": digest(identity), "model": kind, "seed": cfg["seed"]}
                atomic_write(output / "progress.json", {"stage": "train", "model": kind, "seed": cfg["seed"]})
                if (directory / "completed-receipt.json").exists():
                    result, metrics, ablation = completed(directory, run_id)
                else:
                    result = train_one(examples, store, cfg, identity, kind, cfg["seed"], directory)
                    atomic_write(output / "progress.json", {"stage": "validation_and_ablations", "model": kind})
                    metrics, ablation = finish(directory, result, validation, store, cfg, identity, kind)
                results[kind] = {
                    "status": result["status"],
                    "selected_epoch": result["selected_epoch"],
                    "epochs_completed": result["epochs_completed"],
                    "stopping_reason": result["stopping_reason"],
                    "evaluation": metrics,
                    "ablation": ablation,
                }
                gc.collect()
                torch.cuda.empty_cache()
            baseline = read(ROOT / cfg["provenance"]["baseline_summary"]["path"])
            comparison = {
                k: {
                    "baseline_f1": baseline["models"][k]["evaluation"]["constrained"]["micro"]["f1"],
                    "lexical_f1": r["evaluation"]["constrained"]["micro"]["f1"],
                    "baseline_exact": baseline["models"][k]["evaluation"]["constrained"]["exact_target_graph_accuracy"],
                    "lexical_exact": r["evaluation"]["constrained"]["exact_target_graph_accuracy"],
                }
                for k, r in results.items()
            }
            summary = {
                "experiment_sha256": digest(identity),
                "scope": "same compact labels, inputs, split, frozen embeddings, seed, training budget and selection protocol as baseline; new lexical residuals only",
                "models": results,
                "comparison": comparison,
                "test_used": False,
                "automatic_followup_training": False,
            }
            write(output / "summary.json", summary)
            progress = {"stage": "complete_pilot", "summary": str(output / "summary.json"), "comparison": comparison}
            atomic_write(output / "progress.json", progress)
            print(json.dumps(progress), flush=True)
        except BaseException as error:
            atomic_write(
                output / "progress.json",
                {
                    "stage": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                    "error": str(error),
                    "resume": "Rerun identical committed launcher.",
                },
            )
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["plan", "run"], default="plan")
    p.add_argument("--report", type=Path)
    args = p.parse_args()
    cfg, examples, validation, store, identity, report = preflight()
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.report:
        write(args.report, report)
    if args.stage == "run":
        execute(cfg, examples, validation, store, identity, report)


if __name__ == "__main__":
    main()
