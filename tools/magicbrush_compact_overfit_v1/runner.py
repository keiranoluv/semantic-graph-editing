"""Prepare on CPU; user runs frozen CLIP caching then sequential MLP/GNN on CUDA."""
import argparse
import gc
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch

from src.datasets.gqa_full import atomic_write
from src.editing.operations import KINDS, EditGraph, Operation
from tools.edit_model_v1.codec import normalize_label
from tools.edit_model_v1.data import Example
from tools.edit_model_v1.model import collate_inputs
from tools.edit_model_v1.text import text_key
from tools.emnlp2020_graph_modification_v1.dataset import read, sha256, write
from tools.magicbrush_compact_v1.build import CONFIG as COMPACT_CONFIG
from tools.magicbrush_compact_v1.build import example_dict
from tools.magicbrush_compact_v1.build import source_hashes as compact_hashes
from tools.magicbrush_compact_v1.projection import project_example
from tools.magicbrush_experiment_v3.cache import build_cache, input_texts
from tools.magicbrush_experiment_v3.pipeline import preflight as original_preflight
from tools.magicbrush_experiment_v3.pipeline import versions
from tools.magicbrush_experiment_v3.targets import encode_targets
from tools.magicbrush_experiment_v3.vocabulary import build_vocabulary, coverage, prototype_inputs
from tools.magicbrush_overfit_v1.data import example_receipt
from tools.magicbrush_overfit_v1.runner import DEFAULT_CONFIG as OLD_PROBE
from tools.magicbrush_overfit_v1.runner import source_hashes as probe_hashes
from tools.magicbrush_overfit_v1.runner import train_one
from tools.prepare_magicbrush import ROOT, digest
from tools.run_gqa_full import exclusive

CONFIG = ROOT / "configs/level2/magicbrush-compact-overfit16-v1.json"


def source_hashes():
    files = [*Path(__file__).parent.glob("*.py"), ROOT / "scripts/run_magicbrush_compact_overfit16_v1.sh"]
    return {**compact_hashes(), **probe_hashes(), **{str(p.relative_to(ROOT)): sha256(p) for p in sorted(files)}}


def load_train(payload, original_train, policy):
    rows = payload["records"]
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate compact record IDs")
    selected = {r["id"]: r for r in rows if r["split"] == "train"}
    if set(selected) != {e.id for e in original_train}:
        raise ValueError("Compact train membership changed")
    train = []
    for old in original_train:
        row = selected[old.id]
        projected, audit = project_example(old, policy)
        expected = {
            **example_dict(projected),
            "original_label_sha256": digest(example_dict(old)),
            "flags": audit["flags"],
        }
        if digest(row) != digest(expected) or row["flags"]:
            raise ValueError("Compact record differs from frozen policy or has flags: " + old.id)
        source, target = (EditGraph.from_dict(row[k]) for k in ("source_graph", "target_graph"))
        operations = tuple(Operation.from_dict(o) for o in row["operations"])
        source, target, operations = normalize_label(source, target, operations)
        train.append(Example(old.id, "train", old.group, old.instruction, source, target, operations))
    return train


def select_fixed(train, receipt):
    ids = [r["id"] for r in receipt["cases"]]
    index = {e.id: e for e in train}
    if len(ids) != 16 or len(set(ids)) != 16 or not set(ids) <= set(index):
        raise ValueError("Expected exact previous 16 train IDs")
    cohort = [index[i] for i in ids]
    if any(e.split != "train" or not e.operations for e in cohort) or len({e.group for e in cohort}) != 16:
        raise ValueError("Expected nonempty train-only distinct-group cohort")
    if set(o.kind for e in cohort for o in e.operations) != set(KINDS):
        raise ValueError("Fixed compact cohort no longer covers all six operations")
    return cohort


def verify_artifacts(specs):
    paths = {}
    for name, spec in specs.items():
        path = (ROOT / spec["path"]).resolve()
        if not path.is_relative_to(ROOT) or sha256(path) != spec["sha256"]:
            raise ValueError("Pinned artifact changed: " + name)
        paths[name] = path
    return paths


def preflight(path=CONFIG):
    config = read(path)
    if read(path.with_suffix(".lock.json")) != {
        "config_sha256": sha256(path),
        "source_sha256": source_hashes(),
        "versions": versions(),
    }:
        raise ValueError("Compact probe release lock changed")
    paths = verify_artifacts(config["artifacts"])
    compact_cfg = read(COMPACT_CONFIG)
    if read(COMPACT_CONFIG.with_suffix(".lock.json")) != {
        "config_sha256": sha256(COMPACT_CONFIG),
        "source_sha256": compact_hashes(),
    }:
        raise ValueError("Compact label release changed")
    old = read(OLD_PROBE)
    if read(OLD_PROBE.with_suffix(".lock.json")) != {
        "config_sha256": sha256(OLD_PROBE),
        "source_sha256": probe_hashes(),
        "versions": versions(),
    }:
        raise ValueError("Original memorization release changed")
    for k in (
        "models",
        "seed",
        "model",
        "lr",
        "weight_decay",
        "gradient_clip",
        "loss_action_weight",
        "action_class_weights",
        "presence_class_weights",
        "batch_size",
        "val_batch_size",
        "max_steps",
        "evaluate_every",
        "required_success_checks",
    ):
        if config[k] != old[k]:
            raise ValueError("Must match previous probe budget/architecture: " + k)
    if config["schema"] != "magicbrush-compact-overfit16-v1" or config["device"] != "cuda":
        raise ValueError("Unexpected compact probe schema/device")
    if (
        config["output"] != "runs/level2/magicbrush-compact-overfit16-v1"
        or config["text_cache"] != "data/processed/level2/magicbrush-compact-overfit16-v1/clip-train"
    ):
        raise ValueError("Expected isolated new run/cache paths")
    parent, examples, _, _, _ = original_preflight(ROOT / compact_cfg["parent_config"])
    if any(config[k] != parent[k] for k in ("weights", "weights_sha256", "text_batch_size")):
        raise ValueError("Frozen CLIP configuration must match v3")
    train = load_train(read(paths["compact_edits"]), examples["train"], read(paths["compact_policy"]))
    vocab = build_vocabulary(train)
    if vocab != read(paths["compact_vocabulary"]):
        raise ValueError("Compact vocabulary differs from all 301 train labels")
    cohort = select_fixed(train, read(paths["previous_cohort"]))
    receipt = {
        "scope": "same 16 train IDs as overfit16-v1, compact source/target graphs; canonical new IDs recomputed",
        "cases": [example_receipt(e) for e in cohort],
    }
    if read(paths["cohort"]) != receipt:
        raise ValueError("Compact cohort receipt changed")
    cov = coverage(cohort, vocab)
    if cov["oov_payload_fields"]:
        raise ValueError("Probe labels outside compact train vocabulary")
    encode_targets(cohort, vocab, config)  # CPU target shape and budget checks.
    texts = input_texts(cohort + prototype_inputs(vocab))
    identity = {
        "schema": config["schema"],
        "config": config,
        "release_lock_sha256": sha256(path.with_suffix(".lock.json")),
        "vocabulary": vocab,
        "vocabulary_sha256": digest(vocab),
        "cohort_sha256": sha256(paths["cohort"]),
        "texts_sha256": digest([text_key(t) for t in texts]),
        "initialization": "fresh random seed 42, no trained checkpoint",
        "scope": "same 16 compact train examples; no validation/test training, selection or scoring",
    }
    report = {
        "ready": True,
        "gpu_jobs_started": False,
        "cohort_ids": [e.id for e in cohort],
        "cohort_operation_support": dict(Counter(o.kind for e in cohort for o in e.operations)),
        "vocabulary_sizes": {k: len(vocab[k]) for k in ("labels", "attributes", "predicates")},
        "vocabulary_scope": "all 301 compact train examples, not only 16",
        "cohort_coverage": cov,
        "text_count_to_encode": len(texts),
        "clip_batch_size": config["text_batch_size"],
        "gpu_queue": ["frozen CLIP text cache", "MLP seed 42", "GNN seed 42"],
        "training_batch_size": config["batch_size"],
        "max_updates_per_model": config["max_steps"],
        "success": "raw AND constrained operations AND target graph exact 16/16, two successive checks",
        "stop_after_two_models": True,
        "test_labels_decoded": False,
        "validation_used_for_probe_training_selection_scoring": False,
        "parent_note": "Parent integrity checks parse trainval artifacts; probe only uses train examples and train-derived vocabulary.",
        "actual_tensor_collision_check": "performed after user builds frozen GPU cache; not yet asserted by CPU plan",
        "comparison_note": "Same cohort, budget and architecture; annotations, source attributes and attribute vocabulary differ. Not evidence of improved generalization.",
        "experiment_sha256": digest(identity),
    }
    return config, cohort, identity, report


def verify_features(cohort, store):
    inputs = collate_inputs([e.model_input() for e in cohort], store, device="cpu")
    signatures = []
    for i in range(len(cohort)):
        h = hashlib.sha256()
        for name, value in sorted(inputs.items()):
            h.update(name.encode())
            h.update(value[i].contiguous().numpy().tobytes())
        signatures.append(h.hexdigest())
    if len(set(signatures)) != len(cohort):
        raise ValueError("Frozen compact model inputs collide")
    return signatures


def execute(config, cohort, identity, report):
    output = ROOT / config["output"]
    cache = ROOT / config["text_cache"]
    with exclusive(ROOT / "runs/level2/.gqa-kg-training-device"), exclusive(output):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; GPU work belongs to user launcher")
        torch.set_num_threads(1)
        torch.backends.cudnn.benchmark = False
        write(output / "experiment.json", identity)
        write(output / "preflight.json", report)
        try:
            atomic_write(output / "progress.json", {"stage": "cache_compact_train_sources_and_train_vocabulary"})
            # The reused cache API calls this scope trainval; source_input_ids
            # contains exactly the 16 TRAIN IDs, with zero validation examples.
            store, meta = build_cache(cohort, config, identity, cache, ROOT, scope="trainval")
            features = {
                "cache_manifest_sha256": sha256(cache / "manifest.json"),
                "vectors_sha256": meta["vectors_sha256"],
                "model_input_tensor_sha256": verify_features(cohort, store),
            }
            write(output / "feature-receipt.json", features)
            run_identity = {**identity, "frozen_features": features}
            results = {}
            for kind in config["models"]:
                atomic_write(
                    output / "progress.json",
                    {"stage": "train_memorization_probe", "model": kind, "seed": config["seed"]},
                )
                results[kind] = train_one(
                    cohort, store, config, run_identity, kind, output / kind / str(config["seed"])
                )
                gc.collect()
                torch.cuda.empty_cache()
            summary = {
                "scope": identity["scope"],
                "experiment_sha256": digest(identity),
                "training_identity_sha256": digest(run_identity),
                "comparison_note": report["comparison_note"],
                "automatic_followup_training": False,
                "models": {
                    k: {field: r[field] for field in ("status", "selected_step", "steps_completed", "evaluation")}
                    for k, r in results.items()
                },
            }
            write(output / "summary.json", summary)
            progress = {
                "stage": "complete_diagnostic",
                "outcomes": {k: r["status"] for k, r in results.items()},
                "summary": str(output / "summary.json"),
            }
            atomic_write(output / "progress.json", progress)
            print(json.dumps(progress), flush=True)
        except BaseException as error:
            atomic_write(
                output / "progress.json",
                {
                    "stage": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                    "error": str(error),
                    "resume": "Rerun same launcher; verified cache and completed models reused.",
                },
            )
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("plan", "run"), default="plan")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    config, cohort, identity, report = preflight()
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.report:
        write(args.report, report)
    if args.stage == "run":
        execute(config, cohort, identity, report)


if __name__ == "__main__":
    main()
