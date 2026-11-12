"""One-run trainer with validation-only selection, provenance and resumable state."""

import argparse
import copy
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path

import torch
from torch.nn import functional as F

from src.datasets.gqa_full import atomic_write
from src.knowledge.transe.core import sha256
from tools.edit_model_v1.codec import PAD, score_predictions
from tools.edit_model_v1.data import load_examples, review_gate
from tools.edit_model_v1.model import EditPredictor, collate_inputs, collate_targets
from tools.edit_model_v1.text import cache_texts, load_store
from tools.prepare_magicbrush import ROOT, digest, encode, fixed
from tools.run_gqa_full import exclusive

DEFAULT_CONFIG = ROOT / "configs/level2/magicbrush-edit-model-v1.json"


def validate_config(config):
    required = set(json.loads(DEFAULT_CONFIG.read_text()))
    if set(config) != required or config["schema_version"] != 1 or config["device"] != "cuda":
        raise ValueError("Expected graph-edit v1 config with CUDA user-run device")
    if config["models"] != ["mlp", "gnn"] or config["seeds"] != [42, 123, 3407]:
        raise ValueError("Expected matched MLP/GNN and three seeds")
    integer_keys = (
        "minimum_valid_pairs",
        "minimum_reviews",
        "text_batch_size",
        "batch_size",
        "val_batch_size",
        "epochs",
        "patience",
        "max_action_tokens",
    )
    if any(type(config[k]) is not int or config[k] <= 0 for k in integer_keys):
        raise ValueError("Expected positive integer budgets")
    if config["minimum_valid_pairs"] < 300 or config["minimum_reviews"] < 20:
        raise ValueError("Main training requires >=300 usable train+val pairs and >=20 determinate reviews")
    if (
        not 0 < config["minimum_correct_fraction"] <= 1
        or any(config[k] <= 0 for k in ("lr", "gradient_clip"))
        or config["weight_decay"] < 0
    ):
        raise ValueError("Invalid optimization or review policy")
    architecture = config["model"]
    if set(architecture) != {"hidden", "byte_dim", "max_nodes", "max_edges", "layers"} or any(
        type(v) is not int or v <= 0 for v in architecture.values()
    ):
        raise ValueError("Invalid model architecture")
    if architecture["max_nodes"] != 20 or architecture["max_edges"] != 60:
        raise ValueError("v1 source schema budget is fixed at 20 nodes / 60 edges")
    for key, parent in (("text_cache", "data"), ("output", "runs")):
        if not (ROOT / config[key]).resolve().is_relative_to(ROOT / parent):
            raise ValueError(f"{key} must stay under {parent}/")


def source_hashes():
    files = [
        *Path(__file__).parent.glob("*.py"),
        ROOT / "tools/train_magicbrush_edits.py",
        ROOT / "src/editing/operations.py",
        ROOT / "src/editing/evaluation.py",
        ROOT / "tools/prepare_magicbrush.py",
        ROOT / "tools/review_magicbrush.py",
    ]
    result = {str(p.relative_to(ROOT)): sha256(p) for p in sorted(files)}
    spec = importlib.util.find_spec("clip")
    if spec is None or spec.origin is None:
        raise ValueError("Pinned CLIP package is missing")
    for path in sorted(Path(spec.origin).parent.glob("*.py")):
        result[f"installed_clip/{path.name}"] = sha256(path)
    return result


def preflight(config):
    validate_config(config)
    paths = {k: ROOT / config[k] for k in ("dataset_manifest", "edits", "review_packet", "reviews_csv")}
    missing = [k for k, p in paths.items() if not p.is_file()]
    report = {"stage": "CPU_preflight_no_GPU", "ready": False, "missing": missing, "test_used": False}
    if not paths["dataset_manifest"].is_file():
        return {}, report, None
    manifest = json.loads(paths["dataset_manifest"].read_text())
    report["raw_pairs"] = len(manifest["records"])
    gate = None
    if not any(k in missing for k in ("edits", "review_packet", "reviews_csv")):
        gate = review_gate(
            paths["dataset_manifest"],
            paths["edits"],
            paths["review_packet"],
            paths["reviews_csv"],
            minimum=config["minimum_reviews"],
            minimum_correct_fraction=config["minimum_correct_fraction"],
        )
        report["human_review"] = gate
    examples = {}
    if "edits" not in missing:
        examples, report["loader"] = load_examples(
            paths["dataset_manifest"],
            paths["edits"],
            rejected=gate["rejected_ids"] if gate else (),
            max_action_tokens=config["max_action_tokens"],
        )
    counts = {k: len(v) for k, v in examples.items()}
    report["usable_pairs"] = counts
    report["minimum_usable_train_plus_val"] = config["minimum_valid_pairs"]
    report["ready"] = bool(
        not missing
        and gate
        and gate["ready"]
        and counts.get("train", 0) > 0
        and counts.get("val", 0) > 0
        and sum(counts.values()) >= config["minimum_valid_pairs"]
    )
    if not report["ready"]:
        report[
            "reason"
        ] = "Need frozen full labels, >=300 usable train+val pairs, nonempty val and verified human review gate"
        return examples, report, None
    identity = {
        "schema_version": 1,
        "config": config,
        "dataset_manifest_sha256": sha256(paths["dataset_manifest"]),
        "edits_sha256": sha256(paths["edits"]),
        "review": gate,
        "source_sha256": source_hashes(),
        "versions": {k: importlib.metadata.version(k) for k in ("torch", "numpy", "clip")},
        "train_ids": [e.id for e in examples["train"]],
        "val_ids": [e.id for e in examples["val"]],
        "model_inputs": "source graph + frozen CLIP instruction/node/predicate text only",
        "selection": "validation autoregressive micro operation F1; ties keep earlier epoch",
        "test_used": False,
    }
    return examples, report, identity


def save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def capture_rng(device):
    return {
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if str(device).startswith("cuda") else [],
    }


def restore_rng(state, device):
    torch.set_rng_state(state["cpu"].cpu())
    if str(device).startswith("cuda"):
        torch.cuda.set_rng_state_all(state["cuda"])


def training_step(model, optimizer, inputs, previous, expected, gradient_clip):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs, previous)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), expected.reshape(-1), ignore_index=PAD)
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite training loss")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip, error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach()), int((expected != PAD).sum()), float(norm)


def validate(model, examples, store, config, device):
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(examples), config["val_batch_size"]):
            batch = examples[start : start + config["val_batch_size"]]
            inputs = collate_inputs([e.model_input() for e in batch], store, device=device)
            predictions.extend(model.predict(inputs, config["max_action_tokens"]))
    metrics, errors = score_predictions(examples, predictions)
    return metrics, [
        {"id": e.id, "tokens": tokens, "error": error}
        for e, tokens, error in zip(examples, predictions, errors, strict=True)
    ]


def train_one(examples, store, config, identity, kind, seed, output, *, device="cuda", wandb_mode="disabled"):
    """CPU is supported for synthetic unit tests only; CLI main training uses CUDA."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    run_identity = {"experiment_sha256": digest(identity), "model": kind, "seed": seed}
    fixed(output / "identity.json", encode(run_identity))
    if (output / "result.json").exists():
        result = json.loads((output / "result.json").read_text())
        if (
            result["identity"] != run_identity
            or result["best_checkpoint_sha256"] != sha256(output / "best.pt")
            or result["last_checkpoint_sha256"] != sha256(output / "last.pt")
        ):
            raise ValueError("Completed run identity/checkpoint changed")
        return result
    torch.manual_seed(seed)
    model = EditPredictor(kind=kind, text_dim=store.dim, **config["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    epoch, best, bad_epochs, history = 0, -1.0, 0, []
    if (output / "last.pt").exists():
        checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
        if checkpoint["identity"] != run_identity:
            raise ValueError("Resume identity mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch, best, bad_epochs, history = (checkpoint[k] for k in ("epoch", "best", "bad_epochs", "history"))
        restore_rng(checkpoint["rng"], device)
        if not (output / "best.pt").exists():
            raise ValueError("Resume requires the selected best checkpoint")
    tracker = None
    if wandb_mode != "disabled":
        import wandb

        tracker = wandb.init(
            entity="keiranoluv-hcmus",
            project="semantic-graph-editing",
            mode=wandb_mode,
            name=f"magicbrush-edit-{kind}-seed{seed}-bs{config['batch_size']}",
            id=digest(run_identity)[:20],
            resume="allow",
            config={**identity, **run_identity},
            tags=[
                "magicbrush",
                "edit-prediction",
                kind,
                f"seed-{seed}",
                f"batch-{config['batch_size']}",
                "frozen-clip-text",
            ],
        )
    try:
        while epoch < config["epochs"] and bad_epochs < config["patience"]:
            epoch += 1
            order = torch.randperm(
                len(examples["train"]), generator=torch.Generator().manual_seed(seed + epoch)
            ).tolist()
            loss_sum, token_count = 0.0, 0
            for start in range(0, len(order), config["batch_size"]):
                batch = [examples["train"][i] for i in order[start : start + config["batch_size"]]]
                inputs = collate_inputs([e.model_input() for e in batch], store, device=device)
                previous, expected = collate_targets(batch, device)
                loss, tokens, _ = training_step(model, optimizer, inputs, previous, expected, config["gradient_clip"])
                loss_sum += loss * tokens
                token_count += tokens
            metrics, _ = validate(model, examples["val"], store, config, device)
            score = metrics["micro"]["f1"]
            if score is None:
                raise ValueError("Validation requires supported operation labels")
            improved = score > best
            best, bad_epochs = (score, 0) if improved else (best, bad_epochs + 1)
            history.append({"epoch": epoch, "train_token_loss": loss_sum / token_count, "validation": metrics})
            checkpoint = {
                "identity": run_identity,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best": best,
                "bad_epochs": bad_epochs,
                "history": history,
                "rng": capture_rng(device),
            }
            if improved:
                save_checkpoint(output / "best.pt", checkpoint)
            save_checkpoint(output / "last.pt", checkpoint)
            print(
                json.dumps(
                    {
                        "model": kind,
                        "seed": seed,
                        "epoch": epoch,
                        "train_loss": loss_sum / token_count,
                        "val_micro_f1": score,
                    }
                ),
                flush=True,
            )
            if tracker:
                tracker.log(
                    {
                        "epoch": epoch,
                        "train/token_loss": loss_sum / token_count,
                        "val/micro_f1": score,
                        "val/exact": metrics["exact_operation_set_accuracy"],
                        "val/valid": metrics["valid_transaction_rate"],
                    }
                )
        selected = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(selected["model"])
        metrics, predictions = validate(model, examples["val"], store, config, device)
        if metrics["micro"]["f1"] != selected["best"]:
            raise ValueError("Reloaded best checkpoint does not reproduce selection metric")
        # No-edit is a transparent nonlearned baseline on the SAME validation cases.
        from tools.edit_model_v1.codec import encode_actions

        noop, _ = score_predictions(examples["val"], [encode_actions(())[1:] for _ in examples["val"]])
        result = {
            "status": "PASS",
            "scope": "validation_of_pseudo_labels_not_human_GT",
            "identity": run_identity,
            "selected_epoch": selected["epoch"],
            "epochs_completed": epoch,
            "stopping_reason": "early_stopping" if bad_epochs >= config["patience"] else "epoch_budget",
            "validation": metrics,
            "no_edit_baseline_validation": noop,
            "train_examples": len(examples["train"]),
            "val_examples": len(examples["val"]),
            "best_checkpoint_sha256": sha256(output / "best.pt"),
            "last_checkpoint_sha256": sha256(output / "last.pt"),
            "text_encoder_frozen": True,
            "test_used": False,
            "history": history,
        }
        fixed(output / "predictions-val.json", encode(predictions))
        fixed(output / "result.json", encode(result))
        return result
    finally:
        if tracker:
            tracker.finish()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=("plan", "cache-text", "train"), default="plan")
    parser.add_argument("--model", choices=("mlp", "gnn"))
    parser.add_argument("--seed", type=int, choices=(42, 123, 3407))
    parser.add_argument("--reviews-csv", type=Path)
    parser.add_argument("--review-packet", type=Path)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    args = parser.parse_args(argv)
    config = copy.deepcopy(json.loads(args.config.read_text()))
    for key in ("reviews_csv", "review_packet"):
        if getattr(args, key):
            config[key] = str(getattr(args, key))
    examples, report, identity = preflight(config)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.stage == "plan":
        return 0 if report["ready"] else 2
    if not report["ready"]:
        raise SystemExit("BLOCKED: prerequisites incomplete; no GPU work started")
    if args.stage == "train" and (args.model is None or args.seed is None):
        parser.error("train requires an explicit --model and --seed; no automatic queue")
    cache = ROOT / config["text_cache"]
    store = None
    if args.stage == "train":
        store, cache_metadata = load_store(cache, identity)
        if store.dim != 512:
            raise ValueError("Expected 512-dimensional frozen CLIP ViT-B/32 text")
        identity = {
            **identity,
            "text_cache_manifest_sha256": sha256(cache / "manifest.json"),
            "text_vectors_sha256": cache_metadata["vectors_sha256"],
        }
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    with exclusive(ROOT / "runs/level2/.gqa-kg-training-device"):
        if not torch.cuda.is_available():
            raise SystemExit("User-run GPU command requires CUDA")
        torch.set_num_threads(1)
        torch.backends.cudnn.benchmark = False
        if args.stage == "cache-text":
            with exclusive(cache):
                tracker = None
                if args.wandb_mode != "disabled":
                    import wandb

                    tracker = wandb.init(
                        entity="keiranoluv-hcmus",
                        project="semantic-graph-editing",
                        mode=args.wandb_mode,
                        name=f"magicbrush-edit-clip-text-bs{config['text_batch_size']}",
                        tags=["magicbrush", "frozen-clip-text", "feature-cache", f"batch-{config['text_batch_size']}"],
                        config=identity,
                    )
                try:
                    metadata = cache_texts(examples, config, identity, cache)
                    print(
                        json.dumps(
                            {
                                "status": "PASS",
                                "texts": metadata["shape"][0],
                                "truncated_texts": len(metadata["truncated_text_sha256"]),
                            }
                        ),
                        flush=True,
                    )
                    if tracker:
                        tracker.log(
                            {
                                "cached_texts": metadata["shape"][0],
                                "truncated_texts": len(metadata["truncated_text_sha256"]),
                            }
                        )
                finally:
                    if tracker:
                        tracker.finish()
        else:
            output = ROOT / config["output"]
            with exclusive(output):
                fixed(output / "experiment.json", encode(identity))
                atomic_write(output / "preflight.json", report)
                train_one(
                    examples,
                    store,
                    config,
                    identity,
                    args.model,
                    args.seed,
                    output / args.model / str(args.seed),
                    wandb_mode=args.wandb_mode,
                )
    return 0
