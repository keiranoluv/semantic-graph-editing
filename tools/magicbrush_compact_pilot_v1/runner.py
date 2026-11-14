"""User-run CUDA pilot: frozen text cache, MLP/42, GNN/42, validation report, stop."""
import argparse
import gc
import json
from pathlib import Path

import torch

from src.datasets.gqa_full import atomic_write
from tools.emnlp2020_graph_modification_v1.dataset import read, sha256, write
from tools.magicbrush_compact_overfit_v1.runner import load_train
from tools.magicbrush_compact_overfit_v1.runner import source_hashes as parent_hashes
from tools.magicbrush_compact_overfit_v1.runner import verify_artifacts
from tools.magicbrush_compact_v1.build import CONFIG as COMPACT_CONFIG
from tools.magicbrush_compact_v1.build import source_hashes as compact_hashes
from tools.magicbrush_experiment_v1.evaluation import no_edit
from tools.magicbrush_experiment_v3.cache import build_cache, input_texts
from tools.magicbrush_experiment_v3.model import StructuredPredictor
from tools.magicbrush_experiment_v3.pipeline import preflight as original_preflight
from tools.magicbrush_experiment_v3.pipeline import versions
from tools.magicbrush_experiment_v3.runtime import validate
from tools.magicbrush_experiment_v3.targets import encode_targets
from tools.magicbrush_experiment_v3.training import SELECTION_POLICY, train_one
from tools.magicbrush_experiment_v3.vocabulary import build_vocabulary, coverage, prototype_inputs, prototype_vectors
from tools.prepare_magicbrush import ROOT, digest
from tools.run_gqa_full import exclusive

from .data import evaluation_report, load_validation, partition

CONFIG = ROOT / "configs/level2/magicbrush-compact-pilot-v1.json"


def source_hashes():
    files = [*Path(__file__).parent.glob("*.py"), ROOT / "scripts/run_magicbrush_compact_pilot_v1.sh"]
    return {**parent_hashes(), **{str(p.relative_to(ROOT)): sha256(p) for p in sorted(files)}}


def preflight():
    cfg = read(CONFIG)
    if read(CONFIG.with_suffix(".lock.json")) != {
        "config_sha256": sha256(CONFIG),
        "source_sha256": source_hashes(),
        "versions": versions(),
    }:
        raise ValueError("Pilot release changed")
    paths = verify_artifacts(cfg["artifacts"])
    compact = read(COMPACT_CONFIG)
    if read(COMPACT_CONFIG.with_suffix(".lock.json")) != {
        "config_sha256": sha256(COMPACT_CONFIG),
        "source_sha256": compact_hashes(),
    }:
        raise ValueError("Compact release changed")
    parent, original, _, _, _ = original_preflight(ROOT / compact["parent_config"])
    for k in [
        "model",
        "batch_size",
        "val_batch_size",
        "epochs",
        "minimum_epochs",
        "patience",
        "validation_interval",
        "lr",
        "weight_decay",
        "gradient_clip",
        "loss_action_weight",
        "action_class_weights",
        "presence_class_weights",
        "weights",
        "weights_sha256",
        "text_batch_size",
        "quality_gate",
    ]:
        if cfg[k] != parent[k]:
            raise ValueError("Pilot recipe must match full v3: " + k)
    if (
        cfg["schema"] != "magicbrush-compact-pilot-v1"
        or cfg["device"] != "cuda"
        or cfg["models"] != ["mlp", "gnn"]
        or cfg["seed"] != 42
    ):
        raise ValueError("Expected fixed two-model seed-42 CUDA pilot")
    if (
        cfg["output"] != "runs/level2/magicbrush-compact-pilot-v1"
        or cfg["text_cache"] != "data/processed/level2/magicbrush-compact-pilot-v1/clip-trainval"
    ):
        raise ValueError("Expected isolated pilot output/cache")
    prerequisite = read(paths["memorization_summary"])
    if not all(prerequisite["models"][k]["status"] == "MEMORIZED" for k in cfg["models"]):
        raise ValueError("Compact memorization prerequisite not met")
    payload, policy = read(paths["compact_edits"]), read(paths["compact_policy"])
    train = load_train(payload, original["train"], policy)
    validation = load_validation(payload, original["val"], policy)
    scored, gaps = partition(validation)
    if (len(train), len(validation), len(scored)) != (301, 67, 64) or sorted(e.id for e in gaps) != cfg[
        "schema_gap_ids"
    ]:
        raise ValueError("Frozen compact cohort or schema-gap membership changed")
    vocab = build_vocabulary(train)
    if vocab != read(paths["compact_vocabulary"]):
        raise ValueError("Compact train vocabulary changed")
    for examples in (train, scored):
        encode_targets(examples, vocab, cfg)
    texts = input_texts(train + validation + prototype_inputs(vocab))
    identity = {
        "schema": cfg["schema"],
        "config": cfg,
        "release_lock_sha256": sha256(CONFIG.with_suffix(".lock.json")),
        "vocabulary": vocab,
        "vocabulary_sha256": digest(vocab),
        "selection_policy": SELECTION_POLICY,
        "selection_population": "fixed 64 actionable validation pairs; all vocabulary OOV cases retained",
        "train_ids": [e.id for e in train],
        "validation_ids": [e.id for e in validation],
        "scored_validation_ids": [e.id for e in scored],
        "schema_gap_ids": [e.id for e in gaps],
        "initialization": "fresh random seed 42 per model; no overfit or v3 trained checkpoint reused",
        "test_policy": "no test label/source cache/evaluation in this pilot",
    }
    report = {
        "ready": True,
        "gpu_jobs_started": False,
        "train_pairs": len(train),
        "validation_pairs": len(validation),
        "scored_validation_pairs": len(scored),
        "schema_gap_ids": [e.id for e in gaps],
        "gap_policy": "All 67 predictions saved; score 64 actionable cases, report coverage and conservative exact /67; gaps never count as correct empty requests",
        "train_coverage": coverage(train, vocab),
        "scored_validation_vocabulary_coverage": coverage(scored, vocab),
        "vocabulary_sizes": {k: len(vocab[k]) for k in ["labels", "attributes", "predicates"]},
        "text_count": len(texts),
        "text_batch_size": cfg["text_batch_size"],
        "batch_size": cfg["batch_size"],
        "val_batch_size": cfg["val_batch_size"],
        "queue": ["frozen CLIP cache for train/validation sources + train vocabulary", "MLP seed 42", "GNN seed 42"],
        "epochs_max": cfg["epochs"],
        "early_stopping": {
            "minimum_epochs": cfg["minimum_epochs"],
            "patience_epochs": cfg["patience"],
            "validation_interval": cfg["validation_interval"],
            "selection": SELECTION_POLICY,
        },
        "quality_gate": {
            "thresholds": cfg["quality_gate"],
            "behavior": "Report for each model; always run both pilot models. No automatic further seeds or test.",
        },
        "test_labels_decoded": False,
        "automatic_followup_training": False,
        "experiment_sha256": digest(identity),
    }
    return cfg, {"train": train, "val": scored}, validation, identity, report


def verify_completed(directory, run_id):
    receipt = read(directory / "completed-receipt.json")
    names = {
        "best.pt",
        "last.pt",
        "result.json",
        "predictions-val.json",
        "predictions-val-all.json",
        "validation-report.json",
    }
    if receipt["identity"] != run_id or set(receipt["files"]) != names:
        raise ValueError("Pilot completion receipt differs")
    for name, h in receipt["files"].items():
        if sha256(directory / name) != h:
            raise ValueError("Completed pilot artifact modified: " + name)
    return read(directory / "result.json"), read(directory / "validation-report.json")


def finish_validation(directory, result, validation, store, cfg, identity, kind, *, device="cuda"):
    # Reuse the 64 predictions made by checkpoint selection; infer only the
    # three schema-gap inputs for diagnostics, with no correctness claim.
    scored, gaps = partition(validation)
    rows = read(directory / "predictions-val.json")
    if [r["id"] for r in rows] != [e.id for e in scored]:
        raise ValueError("Selected validation IDs changed")
    if gaps:
        saved = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
        model = StructuredPredictor(
            kind=kind,
            vocabulary=identity["vocabulary"],
            prototypes=prototype_vectors(store, identity["vocabulary"]),
            text_dim=store.dim,
            **cfg["model"],
        ).to(device)
        model.load_state_dict(saved["model"])
        _, extra = validate(model, gaps, store, cfg, device)
        rows += extra
        del model
    by_id = {r["id"]: r for r in rows}
    all_rows = [by_id[e.id] for e in validation]
    report = evaluation_report(validation, all_rows)
    if any(report["constrained"][k] != v for k, v in result["validation"].items()):
        raise ValueError("Selected checkpoint metrics disagree with stored predictions")
    if any(report["raw"][k] != v for k, v in result["raw_validation"].items()):
        raise ValueError("Raw metrics disagree")
    write(directory / "predictions-val-all.json", all_rows)
    write(directory / "validation-report.json", report)
    files = [
        "best.pt",
        "last.pt",
        "result.json",
        "predictions-val.json",
        "predictions-val-all.json",
        "validation-report.json",
    ]
    write(
        directory / "completed-receipt.json",
        {"identity": result["identity"], "files": {n: sha256(directory / n) for n in files}},
    )
    return report


def execute(cfg, examples, validation, identity, report):
    output = ROOT / cfg["output"]
    cache = ROOT / cfg["text_cache"]
    with exclusive(ROOT / "runs/level2/.gqa-kg-training-device"), exclusive(output):
        if not torch.cuda.is_available():
            raise RuntimeError("User-run CUDA launcher required")
        torch.set_num_threads(1)
        torch.backends.cudnn.benchmark = False
        write(output / "experiment.json", identity)
        write(output / "preflight.json", report)
        try:
            atomic_write(output / "progress.json", {"stage": "cache_trainval_sources_and_train_vocabulary"})
            store, meta = build_cache(examples["train"] + validation, cfg, identity, cache, ROOT, scope="trainval")
            features = {"manifest_sha256": sha256(cache / "manifest.json"), "vectors_sha256": meta["vectors_sha256"]}
            write(output / "feature-receipt.json", features)
            training_identity = {**identity, "frozen_features": features}
            results = {}
            for kind in cfg["models"]:
                directory = output / kind / str(cfg["seed"])
                run_id = {"experiment_sha256": digest(training_identity), "model": kind, "seed": cfg["seed"]}
                atomic_write(output / "progress.json", {"stage": "train_pilot", "model": kind, "seed": cfg["seed"]})
                if (directory / "completed-receipt.json").exists():
                    result, metrics = verify_completed(directory, run_id)
                else:
                    result = train_one(examples, store, cfg, training_identity, kind, cfg["seed"], directory)
                    metrics = finish_validation(directory, result, validation, store, cfg, training_identity, kind)
                results[kind] = {
                    "status": result["status"],
                    "selected_epoch": result["selected_epoch"],
                    "epochs_completed": result["epochs_completed"],
                    "stopping_reason": result["stopping_reason"],
                    "quality_gate": result["quality_gate"],
                    "evaluation": metrics,
                }
                gc.collect()
                torch.cuda.empty_cache()
            summary = {
                "scope": "Single-seed train/validation compact pilot; not final three-seed/test evidence",
                "experiment_sha256": digest(identity),
                "training_identity_sha256": digest(training_identity),
                "models": results,
                "no_edit_baseline_on_actionable_validation": no_edit(examples["val"]),
                "test_used": False,
                "automatic_followup_training": False,
                "comparison_note": "Compact metrics on 64 actionable validation cases; original v3 used different labels and 67 cases. Do not attribute score changes solely to model improvement.",
            }
            write(output / "summary.json", summary)
            progress = {
                "stage": "complete_pilot",
                "summary": str(output / "summary.json"),
                "gates": {k: r["quality_gate"]["passed"] for k, r in results.items()},
            }
            atomic_write(output / "progress.json", progress)
            print(json.dumps(progress), flush=True)
        except BaseException as error:
            atomic_write(
                output / "progress.json",
                {
                    "stage": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                    "error": str(error),
                    "resume": "Rerun same launcher; resume epoch checkpoint and verify completed models.",
                },
            )
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["plan", "run"], default="plan")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    cfg, examples, validation, identity, report = preflight()
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.report:
        write(args.report, report)
    if args.stage == "run":
        execute(cfg, examples, validation, identity, report)


if __name__ == "__main__":
    main()
