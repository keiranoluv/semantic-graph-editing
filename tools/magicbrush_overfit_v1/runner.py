"""User-run CUDA memorization probe; CPU entrypoint only prepares/verifies it."""
import argparse
import gc
import json
import math
from collections import Counter
from pathlib import Path

import torch

from src.datasets.gqa_full import atomic_write
from tools.edit_model_v1.codec import canonical_actions, decode_actions
from tools.edit_model_v1.model import collate_inputs
from tools.edit_model_v1.runner import capture_rng, restore_rng, save_checkpoint
from tools.emnlp2020_graph_modification_v1.dataset import read, sha256, write
from tools.magicbrush_experiment_v1.evaluation import score_rows
from tools.magicbrush_experiment_v3.model import StructuredPredictor
from tools.magicbrush_experiment_v3.pipeline import preflight as parent_preflight
from tools.magicbrush_experiment_v3.pipeline import source_hashes as parent_hashes
from tools.magicbrush_experiment_v3.pipeline import versions
from tools.magicbrush_experiment_v3.runtime import training_step, validate, validation_loss
from tools.magicbrush_experiment_v3.targets import encode_targets
from tools.magicbrush_experiment_v3.vocabulary import coverage, prototype_vectors
from tools.prepare_magicbrush import ROOT, digest
from tools.run_gqa_full import exclusive

from .data import example_receipt, reuse_store, select_train

DEFAULT_CONFIG = ROOT / "configs/level2/magicbrush-overfit16-v1.json"


def source_hashes():
    result = parent_hashes()
    files = [*Path(__file__).parent.glob("*.py"), ROOT / "scripts/run_magicbrush_overfit16_v1.sh"]
    result.update({str(p.relative_to(ROOT)): sha256(p) for p in sorted(files)})
    return result


def preflight(path=DEFAULT_CONFIG):
    config = read(path)
    lock = read(path.with_suffix(".lock.json"))
    if lock != {"config_sha256": sha256(path), "source_sha256": source_hashes(), "versions": versions()}:
        raise ValueError("Probe release lock differs from code/config/environment")
    if (
        config["schema"] != "magicbrush-overfit16-v1"
        or config["device"] != "cuda"
        or config["models"] != ["mlp", "gnn"]
        or config["seed"] != 42
    ):
        raise ValueError("Expected fixed sequential MLP/GNN CUDA probe")
    for key in [
        "max_steps",
        "evaluate_every",
        "required_success_checks",
        "batch_size",
        "val_batch_size",
        "cohort_size",
    ]:
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError("Invalid budget " + key)
    if config["cohort_size"] != 16 or config["batch_size"] != 16 or config["val_batch_size"] != 16:
        raise ValueError("Production probe uses one full batch of 16 train examples")
    if config["lr"] <= 0 or config["weight_decay"] != 0 or config["model"]["dropout"] != 0:
        raise ValueError("Expected no dropout/weight decay and positive learning rate")
    output = (ROOT / config["output"]).resolve()
    if not output.is_relative_to(ROOT / "runs/level2") or "magicbrush-overfit16-v1" not in output.parts:
        raise ValueError("Probe must use isolated output")
    parent_path = ROOT / config["parent_config"]
    if sha256(parent_path) != config["parent_config_sha256"]:
        raise ValueError("Parent config changed")
    parent, examples, _, parent_identity, _ = parent_preflight(parent_path)
    expected_arch = {**parent["model"], "dropout": 0.0}
    if config["model"] != expected_arch or any(
        config[k] != parent[k]
        for k in ["lr", "gradient_clip", "loss_action_weight", "action_class_weights", "presence_class_weights"]
    ):
        raise ValueError(
            "Probe must preserve v3 architecture, learning rate and loss apart from declared regularization changes"
        )
    cohort, selection = select_train(examples["train"], config["cohort_size"], config["selection_seed"])
    cp = ROOT / config["cohort"]
    if sha256(cp) != config["cohort_sha256"]:
        raise ValueError("Pinned cohort changed")
    expected = {
        "scope": "16 train pairs only; no validation/test selection or scoring",
        "parent_config_sha256": sha256(parent_path),
        "selection": selection,
        "cases": [example_receipt(e) for e in cohort],
    }
    if read(cp) != expected:
        raise ValueError("Pinned cohort no longer matches source labels or deterministic selection")
    cache = ROOT / parent["text_cache"]
    if sha256(cache / "manifest.json") != config["cache_manifest_sha256"]:
        raise ValueError("Frozen v3 feature manifest changed")
    store, meta, signatures = reuse_store(cache, parent_identity, cohort, parent_identity["vocabulary"])
    cov = coverage(cohort, parent_identity["vocabulary"])
    if cov["oov_payload_fields"]:
        raise ValueError("Probe cohort has unsupported labels")
    encode_targets(cohort, parent_identity["vocabulary"], config)  # CPU shape/budget checks only.
    identity = {
        "schema": config["schema"],
        "config": config,
        "release_lock_sha256": sha256(path.with_suffix(".lock.json")),
        "vocabulary": parent_identity["vocabulary"],
        "vocabulary_sha256": parent_identity["vocabulary_sha256"],
        "parent_identity_sha256": digest(parent_identity),
        "cohort_sha256": sha256(cp),
        "cache_manifest_sha256": sha256(cache / "manifest.json"),
        "cache_vectors_sha256": meta["vectors_sha256"],
        "model_input_tensor_sha256": signatures,
        "evaluation_scope": "same 16 training pairs; not validation/test performance",
        "selection": "joint raw/constrained memorization, then F1 and lower training loss",
        "initialization": "fresh random seed 42, no trained checkpoint loaded",
    }
    report = {
        "ready": True,
        "gpu_jobs_started": False,
        "cohort_ids": [e.id for e in cohort],
        "selection": selection,
        "parent_release_verified": True,
        "full_train_vocabulary_preserved": True,
        "vocabulary_sizes": {k: len(parent_identity["vocabulary"][k]) for k in ["labels", "attributes", "predicates"]},
        "cohort_vocabulary_coverage": cov,
        "distinct_actual_tensor_inputs": len(set(signatures)),
        "evaluation_scope": identity["evaluation_scope"],
        "test_labels_decoded": False,
        "validation_used_for_probe_training_selection_or_scoring": False,
        "feature_note": "Reuse verified frozen v3 bank; only selected train inputs and train-derived prototypes copied into probe store. Parent preflight also verifies trainval artifacts.",
        "queue": [{"model": k, "seed": config["seed"]} for k in config["models"]],
        "automatic_queue": "always both models sequentially; no full dataset or test queue follows",
        "max_optimizer_updates_per_model": config["max_steps"],
        "success": "100% operation-set AND target-graph exact match for raw AND constrained outputs on two successive checks",
        "experiment_sha256": digest(identity),
    }
    return config, cohort, store, identity, report


def successful(evaluation):
    return all(
        evaluation[side][metric] == 1.0
        for side in ["constrained", "raw"]
        for metric in ["exact_operation_set_accuracy", "exact_target_graph_accuracy"]
    )


def rank(evaluation):
    raw, main = evaluation["raw"], evaluation["constrained"]
    values = (
        min(raw["exact_target_graph_accuracy"], main["exact_target_graph_accuracy"]),
        main["exact_operation_set_accuracy"],
        raw["micro"]["f1"],
        main["micro"]["f1"],
        -evaluation["loss"],
    )
    if any(v is None or not math.isfinite(v) for v in values):
        raise ValueError("Invalid probe selection metric")
    return values


def evaluate(model, cohort, store, config, device, *, heads=False):
    metrics, rows = validate(model, cohort, store, config, device)
    constrained = score_rows(cohort, rows)
    raw = score_rows(cohort, [{"id": r["id"], "tokens": r["raw_tokens"], "error": r["raw_error"]} for r in rows])
    result = {"constrained": constrained, "raw": raw, "loss": validation_loss(model, cohort, store, config, device)}
    if heads:
        inputs = collate_inputs([e.model_input() for e in cohort], store, device=device)
        with torch.inference_mode():
            logits = model(inputs)
        targets = encode_targets(cohort, model.vocabulary, config, device)
        diagnostics = {}
        for field, scores in logits.items():
            expected = targets[field]
            mask = expected != -100
            total = int(mask.sum())
            if field.endswith("_attributes"):
                pred = scores > 0
                positive = expected > 0
                tp = int((pred & positive & mask).sum())
                fp = int((pred & ~positive & mask).sum())
                fn = int((~pred & positive & mask).sum())
                diagnostics[field] = {
                    "known_decisions": total,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                }
            else:
                pred = scores.argmax(-1)
                correct = int(((pred == expected) & mask).sum())
                diagnostics[field] = {
                    "known_decisions": total,
                    "correct": correct,
                    "accuracy": correct / total if total else None,
                }
                if field.endswith(("_action", "_present")):
                    changed = mask & (expected > 0)
                    n = int(changed.sum())
                    diagnostics[field]["positive_decisions"] = n
                    diagnostics[field]["positive_accuracy"] = (
                        int(((pred == expected) & changed).sum()) / n if n else None
                    )
        result["target_conditioned_heads"] = {
            "warning": "Diagnostic head accuracy uses reference supervision masks; generated predictions do not use targets. COPY and equivalent literal choices can differ categorically while being semantically equal. Attribute binary F1 is before count/top-k selection.",
            "fields": diagnostics,
        }
        counts = Counter()
        for e, row in zip(cohort, rows, strict=True):
            try:
                pred = canonical_actions(e.source, decode_actions(row["tokens"], e.source))
            except (ValueError, TypeError, KeyError):
                pred = ()
            keys = {(o.kind, o.entity_id): o for o in pred if not o.kind.startswith("add_")}
            for op in e.operations:
                if op.kind.startswith("add_"):
                    continue
                counts["expected_existing_ops"] += 1
                p = keys.get((op.kind, op.entity_id))
                counts["correct_kind_and_source_id"] += p is not None
                counts["correct_complete_existing_op"] += p == op
                if p is not None and op.kind == "replace_node":
                    counts["matched_replace_node"] += 1
                    counts["matched_replace_node_label_correct"] += p.after.label == op.after.label
                    counts["matched_replace_node_attributes_correct"] += p.after.attributes == op.after.attributes
        result["decoded_source_id_and_payload_counts"] = dict(counts)
    return result, rows


def verify_completed(directory, identity):
    receipt = read(directory / "completed-receipt.json")
    if receipt["identity"] != identity or set(receipt["files"]) != {
        "best.pt",
        "last.pt",
        "result.json",
        "predictions-train.json",
    }:
        raise ValueError("Invalid completed probe receipt")
    for name, checksum in receipt["files"].items():
        if sha256(directory / name) != checksum:
            raise ValueError("Completed probe artifact modified")
    return read(directory / "result.json")


def train_one(cohort, store, config, identity, kind, directory, *, device="cuda"):
    directory.mkdir(parents=True, exist_ok=True)
    run_id = {"experiment_sha256": digest(identity), "model": kind, "seed": config["seed"]}
    write(directory / "identity.json", run_id)
    if (directory / "completed-receipt.json").exists():
        return verify_completed(directory, run_id)
    torch.manual_seed(config["seed"])
    model = StructuredPredictor(
        kind=kind,
        vocabulary=identity["vocabulary"],
        prototypes=prototype_vectors(store, identity["vocabulary"]),
        text_dim=store.dim,
        **config["model"],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.0)
    step, streak, best, history = 0, 0, None, []
    if (directory / "last.pt").exists():
        saved = torch.load(directory / "last.pt", map_location="cpu", weights_only=True)
        if saved["identity"] != run_id:
            raise ValueError("Probe resume identity mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        step, streak, best, history = (saved[k] for k in ["step", "streak", "best", "history"])
        restore_rng(saved["rng"], device)

    def checkpoint_check():
        nonlocal best, streak
        measured, _ = evaluate(model, cohort, store, config, device)
        key = rank(measured)
        streak = streak + 1 if successful(measured) else 0
        if best is None or key > tuple(best["rank"]):
            best = {
                "step": step,
                "rank": key,
                "evaluation": measured,
                "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
        history.append(
            {
                "step": step,
                "loss": measured["loss"],
                "raw_f1": measured["raw"]["micro"]["f1"],
                "raw_exact": measured["raw"]["exact_operation_set_accuracy"],
                "constrained_f1": measured["constrained"]["micro"]["f1"],
                "constrained_exact": measured["constrained"]["exact_operation_set_accuracy"],
                "success_streak": streak,
            }
        )
        save_checkpoint(
            directory / "last.pt",
            {
                "identity": run_id,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "streak": streak,
                "best": best,
                "history": history,
                "rng": capture_rng(device),
            },
        )
        print(json.dumps({"model": kind, **history[-1]}), flush=True)

    if best is None:
        checkpoint_check()
    while step < config["max_steps"] and streak < config["required_success_checks"]:
        # One full 16-example batch = one update; no epoch/patience ambiguity.
        training_step(model, optimizer, cohort, store, config, device)
        step += 1
        if step % config["evaluate_every"] == 0 or step == config["max_steps"]:
            checkpoint_check()
    model.load_state_dict(best["model"])
    final, rows = evaluate(model, cohort, store, config, device, heads=True)
    if any(final[k] != best["evaluation"][k] for k in ["constrained", "raw"]):
        raise ValueError("Selected checkpoint metrics changed")
    save_checkpoint(directory / "best.pt", {"identity": run_id, "step": best["step"], "model": best["model"]})
    write(directory / "predictions-train.json", rows)
    status = "MEMORIZED" if streak >= config["required_success_checks"] else "NOT_MEMORIZED_WITHIN_BUDGET"
    result = {
        "identity": run_id,
        "status": status,
        "scope": "same training cohort, not generalization evidence",
        "steps_completed": step,
        "selected_step": best["step"],
        "stopping_reason": "two_successive_perfect_checks" if status == "MEMORIZED" else "optimizer_update_budget",
        "evaluation": final,
        "history": history,
        "validation_or_test_used": False,
        "best_checkpoint_sha256": sha256(directory / "best.pt"),
        "last_checkpoint_sha256": sha256(directory / "last.pt"),
    }
    write(directory / "result.json", result)
    write(
        directory / "completed-receipt.json",
        {
            "identity": run_id,
            "files": {
                n: sha256(directory / n) for n in ["best.pt", "last.pt", "result.json", "predictions-train.json"]
            },
        },
    )
    return result


def execute(config, cohort, store, identity, report):
    output = ROOT / config["output"]
    with exclusive(ROOT / "runs/level2/.gqa-kg-training-device"), exclusive(output):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA run belongs to the user")
        torch.set_num_threads(1)
        torch.backends.cudnn.benchmark = False
        write(output / "experiment.json", identity)
        write(output / "preflight.json", report)
        results = {}
        try:
            for kind in config["models"]:
                atomic_write(
                    output / "progress.json",
                    {"stage": "train_memorization_probe", "model": kind, "seed": config["seed"]},
                )
                results[kind] = train_one(cohort, store, config, identity, kind, output / kind / str(config["seed"]))
                gc.collect()
                torch.cuda.empty_cache()
            summary = {
                "scope": "same 16 train examples; no validation/test evaluation; not a main experiment",
                "experiment_sha256": digest(identity),
                "models": {
                    k: {
                        "status": r["status"],
                        "selected_step": r["selected_step"],
                        "steps_completed": r["steps_completed"],
                        "evaluation": r["evaluation"],
                    }
                    for k, r in results.items()
                },
                "next_step": "If memorized, investigate consistency/generalization; if not, inspect head diagnostics and optimization. Failure within a finite budget alone does not prove a code defect.",
                "automatic_followup_training": False,
            }
            write(output / "summary.json", summary)
            progress = {
                "stage": "complete_diagnostic",
                "outcomes": {k: r["status"] for k, r in results.items()},
                "summary": str(output / "summary.json"),
            }
            atomic_write(output / "progress.json", progress)
            print(json.dumps(progress), flush=True)
        except BaseException as exc:
            atomic_write(
                output / "progress.json",
                {
                    "stage": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                    "error": str(exc),
                    "resume": "Rerun same command; resume at saved check, verify/skip completed models.",
                },
            )
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=["plan", "run"], default="plan")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    config, cohort, store, identity, report = preflight(args.config.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.report:
        write(args.report, report)
    if args.stage == "run":
        execute(config, cohort, store, identity, report)


if __name__ == "__main__":
    main()
