"""Resumable structured training with train-only semantic prototypes."""

import json
import math
from pathlib import Path

import torch
from common.io import read, sha256, write
from common.training.random_state import capture_rng, restore_rng, save_checkpoint
from task2_graph_editing.magicbrush.preprocess import digest
from task2_graph_editing.magicbrush.sequence_model.codec import encode_actions, score_predictions
from task2_graph_editing.magicbrush.structured_model.model import StructuredPredictor
from task2_graph_editing.magicbrush.structured_model.runtime import training_step, validate, validation_loss
from task2_graph_editing.magicbrush.structured_model.vocabulary import prototype_vectors

SELECTION_POLICY = (
    "validation micro operation F1, then exact target graph accuracy, then lower "
    "validation structured loss (OOV payload fields masked, examples retained); exact ties keep earlier epoch"
)


def quality_gate(metrics, config):
    thresholds = config["quality_gate"]
    valid = metrics["valid_transaction_rate"]
    f1 = metrics["micro"]["f1"]
    passed = (
        valid is not None
        and f1 is not None
        and math.isfinite(valid)
        and math.isfinite(f1)
        and valid >= thresholds["minimum_valid_transaction_rate"]
        and f1 >= thresholds["minimum_micro_f1"]
    )
    return {
        "passed": passed,
        "validation_valid_transaction_rate": valid,
        "validation_micro_f1": f1,
        "thresholds": thresholds,
        "scope": "validation only; operational sanity check, not proof of useful generalization",
    }


def selection_rank(metrics, token_loss):
    rank = (metrics["micro"]["f1"], metrics["exact_target_graph_accuracy"], -token_loss)
    if any(value is None or not math.isfinite(value) for value in rank):
        raise ValueError("Nonfinite or undefined validation selection metric")
    return rank


def should_stop(epoch, last_improved_epoch, config):
    # Patience is in epochs, not validation evaluations. The warmup does not
    # require train loss to converge or test labels to be opened.
    return epoch >= config["minimum_epochs"] and epoch - last_improved_epoch >= config["patience"]


def train_one(examples, store, config, identity, kind, seed, output, *, device="cuda", wandb_mode="disabled"):
    """Only synthetic tests run on CPU; production GPU launch belongs to user."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    run_identity = {"experiment_sha256": digest(identity), "model": kind, "seed": seed}
    write(output / "identity.json", run_identity)
    if (output / "result.json").exists():
        result = read(output / "result.json")
        if (
            result["identity"] != run_identity
            or result["best_checkpoint_sha256"] != sha256(output / "best.pt")
            or result["last_checkpoint_sha256"] != sha256(output / "last.pt")
            or result["predictions_sha256"] != sha256(output / "predictions-val.json")
        ):
            raise ValueError("Completed run identity/checkpoint/predictions changed")
        return result
    torch.manual_seed(seed)
    model = StructuredPredictor(
        kind=kind,
        vocabulary=identity["vocabulary"],
        prototypes=prototype_vectors(store, identity["vocabulary"]),
        text_dim=store.dim,
        **config["model"],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    epoch, last_improved, history, best = 0, 0, [], None
    if (output / "last.pt").exists():
        saved = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
        if saved["identity"] != run_identity:
            raise ValueError("Resume identity mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        epoch, last_improved, history, best = (
            saved[k] for k in ("epoch", "last_improved_epoch", "history", "selected")
        )
        restore_rng(saved["rng"], device)
    tracker = None
    if wandb_mode != "disabled":
        import wandb

        tracker = wandb.init(
            project="semantic-graph-editing",
            mode=wandb_mode,
            name=f"magicbrush-v3-{kind}-seed{seed}-bs{config['batch_size']}",
            id=digest(run_identity)[:20],
            resume="allow",
            config={**identity, **run_identity},
            tags=["magicbrush", "experiment-v3", kind, f"seed-{seed}"],
        )
    try:
        while epoch < config["epochs"] and not should_stop(epoch, last_improved, config):
            epoch += 1
            order = torch.randperm(
                len(examples["train"]), generator=torch.Generator().manual_seed(seed + epoch)
            ).tolist()
            total, count = 0.0, 0
            for start in range(0, len(order), config["batch_size"]):
                batch = [examples["train"][i] for i in order[start : start + config["batch_size"]]]
                loss, tokens = training_step(model, optimizer, batch, store, config, device)
                total += loss * tokens
                count += tokens
            row = {"epoch": epoch, "train_structured_loss": total / count}
            # Always evaluate at the final budget and at the minimum budget.
            if epoch % config["validation_interval"] == 0 or epoch in (1, config["minimum_epochs"], config["epochs"]):
                metrics, _ = validate(model, examples["val"], store, config, device)
                val_loss = validation_loss(model, examples["val"], store, config, device)
                rank = selection_rank(metrics, val_loss)
                if best is None or rank > tuple(best["rank"]):
                    last_improved = epoch
                    best = {
                        "epoch": epoch,
                        "rank": rank,
                        "validation": metrics,
                        "validation_structured_loss": val_loss,
                        "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    }
                row.update(validation=metrics, validation_structured_loss=val_loss)
            history.append(row)
            # Selected weights and optimizer/RNG progress are one atomic file;
            # interruption cannot mix best.pt from one epoch with another last.pt.
            save_checkpoint(
                output / "last.pt",
                {
                    "identity": run_identity,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "last_improved_epoch": last_improved,
                    "history": history,
                    "selected": best,
                    "rng": capture_rng(device),
                },
            )
            log = {"model": kind, "seed": seed, "epoch": epoch, "train_loss": row["train_structured_loss"]}
            if "validation" in row:
                log.update(
                    val_micro_f1=row["validation"]["micro"]["f1"],
                    val_valid=row["validation"]["valid_transaction_rate"],
                    val_structured_loss=row["validation_structured_loss"],
                    selected_epoch=best["epoch"],
                )
            print(json.dumps(log), flush=True)
            if tracker:
                tracker.log(log)
        if best is None:
            raise ValueError("No checkpoint selected")
        model.load_state_dict(best["model"])
        metrics, predictions = validate(model, examples["val"], store, config, device)
        if metrics != best["validation"]:
            raise ValueError("Selected weights do not reproduce saved validation metrics")
        save_checkpoint(
            output / "best.pt",
            {
                "identity": run_identity,
                "model": best["model"],
                "epoch": best["epoch"],
                "rank": best["rank"],
                "validation": metrics,
                "selection_policy": SELECTION_POLICY,
            },
        )
        raw_metrics, _ = score_predictions(examples["val"], [r["raw_tokens"] for r in predictions])
        gate = quality_gate(metrics, config)
        noop, _ = score_predictions(examples["val"], [encode_actions(())[1:] for _ in examples["val"]])
        write(output / "predictions-val.json", predictions)
        result = {
            "status": "VALIDATION_GATE_PASSED" if gate["passed"] else "QUALITY_GATE_FAILED",
            "scope": "validation of frozen AI-assisted labels; not all labels are human GT",
            "identity": run_identity,
            "selected_epoch": best["epoch"],
            "epochs_completed": epoch,
            "stopping_reason": "early_stopping" if should_stop(epoch, last_improved, config) else "epoch_budget",
            "selection_policy": SELECTION_POLICY,
            "validation_structured_loss": best["validation_structured_loss"],
            "validation": metrics,
            "quality_gate": gate,
            "raw_validation": raw_metrics,
            "decoding": "categorical fields and source-copy; constrained and raw metrics both reported",
            "no_edit_baseline_validation": noop,
            "train_examples": len(examples["train"]),
            "val_examples": len(examples["val"]),
            "best_checkpoint_sha256": sha256(output / "best.pt"),
            "last_checkpoint_sha256": sha256(output / "last.pt"),
            "predictions_sha256": sha256(output / "predictions-val.json"),
            "text_encoder_frozen": True,
            "test_used": False,
            "history": history,
        }
        write(output / "result.json", result)
        return result
    finally:
        if tracker:
            tracker.finish()
