"""Epoch-boundary resume and loss-based early stopping; no test labels loaded."""

import json
import math
import time
from pathlib import Path

import torch
from common.io import read, sha256, write
from common.training.random_state import capture_rng, restore_rng, save_checkpoint
from task2_graph_editing.he2020.models.runner import evaluate, make_model, predict, tensor_subset
from task2_graph_editing.he2020.training import tracking
from torch import nn


def initial_state():
    return {"epoch": 0, "best_loss": None, "best_epoch": 0, "anchor_loss": None, "bad_epochs": 0}


def advance(state, loss, config):
    if not math.isfinite(loss):
        raise ValueError("nonfinite_validation_loss")
    state = {**state, "epoch": state["epoch"] + 1}
    selected = state["best_loss"] is None or loss < state["best_loss"]
    if selected:
        state.update(best_loss=loss, best_epoch=state["epoch"])
    if state["anchor_loss"] is None or loss < state["anchor_loss"] - config["min_delta"]:
        state.update(anchor_loss=loss, bad_epochs=0)
    else:
        state["bad_epochs"] += 1
    return state, selected


def should_stop(state, config):
    return state["epoch"] >= config["minimum_epochs"] and state["bad_epochs"] >= config["patience"]


def epoch_pass(model, tensors, targets, config, *, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total = count = 0
    if training:
        chunks = torch.randperm(len(targets[0])).split(config["batch_size"])
    else:
        chunks = [slice(s, s + config["batch_size"]) for s in range(0, len(targets[0]), config["batch_size"])]
    with torch.set_grad_enabled(training):
        for indices in chunks:
            previous, truth = (t[indices].to(config["device"]) for t in targets)
            logits = model(tensor_subset(tensors, indices, config["device"]), previous)
            loss_sum = nn.functional.cross_entropy(
                logits.flatten(0, 1), truth.flatten(), ignore_index=0, reduction="sum"
            )
            if not torch.isfinite(loss_sum):
                raise ValueError("nonfinite_token_loss")
            tokens = int((truth != 0).sum())
            if training:
                optimizer.zero_grad(set_to_none=True)
                (loss_sum / tokens).backward()
                nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
            total += float(loss_sum.detach())
            count += tokens
    return total / count


def checked_result(directory, run_identity):
    result = read(directory / "result.json")
    receipt = read(directory / "completed-receipt.json")
    if result["run_identity"] != run_identity:
        raise ValueError("completed_run_identity_changed")
    for name, checksum in receipt["files"].items():
        if sha256(directory / name) != checksum:
            raise ValueError("completed_run_artifact_changed")
    return result


def train_one(config, kind, seed, examples, tensors, vocabulary, experiment_identity):
    directory = Path(config["output"]) / f"{kind}-seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    run_identity = {"experiment_identity": experiment_identity, "model": kind, "seed": seed}
    write(directory / "identity.json", run_identity)
    if (directory / "completed-receipt.json").exists():
        return checked_result(directory, run_identity)
    torch.manual_seed(seed)
    if config["device"] == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = make_model(config, kind, vocabulary)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    targets = {split: vocabulary.collate(rows) for split, rows in examples.items()}
    state, history, best_state = initial_state(), [], None
    elapsed = 0.0
    if (directory / "last.pt").exists():
        saved = torch.load(directory / "last.pt", weights_only=True, map_location="cpu")
        if saved["run_identity"] != run_identity:
            raise ValueError("resume_identity_changed")
        model.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        state, history, best_state, elapsed = (
            saved[k] for k in ("stop_state", "history", "best_state_dict", "elapsed_seconds")
        )
        restore_rng(saved["rng"], config["device"])
    # Isolate any SDK initialization from model/shuffle randomness.
    rng = capture_rng(config["device"])
    tracker = tracking.start(config, run_identity)
    restore_rng(rng, config["device"])
    started = time.monotonic()
    successful = False
    try:
        if tracker is not None:
            write(directory / "wandb-run.json", {"id": tracker.id, "url": tracker.url, "mode": "online"})
            tracking.log_history(tracker, history)
        while state["epoch"] < config["epochs"] and not should_stop(state, config):
            train_loss = epoch_pass(model, tensors["train"], targets["train"], config, optimizer=optimizer)
            val_loss = epoch_pass(model, tensors["val"], targets["val"], config)
            state, selected = advance(state, val_loss, config)
            if selected:
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            metrics, _, _ = evaluate(examples["val"], predict(model, tensors["val"], vocabulary, config), vocabulary)
            row = {
                "epoch": state["epoch"],
                "train/loss": train_loss,
                "val/loss": val_loss,
                "train/lr": optimizer.param_groups[0]["lr"],
                "early_stop/bad_epochs": state["bad_epochs"],
                "early_stop/best_loss": state["best_loss"],
                "early_stop/best_epoch": state["best_epoch"],
                **tracking.flatten(metrics, "val"),
            }
            if config["device"] == "cuda":
                row["system/peak_memory_mib"] = torch.cuda.max_memory_allocated() / 1024**2
            history.append(row)
            save_checkpoint(
                directory / "last.pt",
                {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "rng": capture_rng(config["device"]),
                    "run_identity": run_identity,
                    "stop_state": state,
                    "history": history,
                    "best_state_dict": best_state,
                    "elapsed_seconds": elapsed + time.monotonic() - started,
                },
            )
            tracking.log_history(tracker, [row])
            print(json.dumps({"model": kind, "seed": seed, **row}), flush=True)
        if best_state is None:
            raise ValueError("missing_selected_checkpoint")
        model.load_state_dict(best_state)
        checkpoint = {
            "state_dict": best_state,
            "epoch": state["best_epoch"],
            "vocabulary": vocabulary.tokens,
            "config": config,
            "experiment_identity": experiment_identity,
        }
        save_checkpoint(directory / "best.pt", checkpoint)
        metrics, cases, selected = evaluate(
            examples["val"], predict(model, tensors["val"], vocabulary, config), vocabulary
        )
        write(directory / "val-predictions.json", cases)
        write(directory / "val-cases.json", selected)
        result = {
            "model": kind,
            "seed": seed,
            "device": config["device"],
            "run_identity": run_identity,
            "best_epoch": state["best_epoch"],
            "best_val_loss": state["best_loss"],
            "epochs_completed": state["epoch"],
            "stopping_reason": "early_stopping" if should_stop(state, config) else "epoch_budget",
            "history": history,
            "validation": metrics,
            "test_used": False,
            "checkpoint_sha256": sha256(directory / "best.pt"),
            "last_checkpoint_sha256": sha256(directory / "last.pt"),
            "runtime_seconds": elapsed + time.monotonic() - started,
            "wandb": read(directory / "wandb-run.json") if tracker is not None else None,
        }
        if tracker is not None:
            tracker.summary.update(
                {
                    "best_epoch": state["best_epoch"],
                    "epochs_completed": state["epoch"],
                    "stopping_reason": result["stopping_reason"],
                    "best_val_loss": state["best_loss"],
                    **tracking.flatten(metrics, "selected_val"),
                }
            )
            tracker.finish(exit_code=0)
            tracker = None
        # Only declare completion after W&B finished successfully.
        if (directory / "result.json").exists() and not (directory / "completed-receipt.json").exists():
            # Recover a crash between immutable result and completion receipt.
            previous = read(directory / "result.json")
            for key in ("runtime_seconds",):
                result[key] = previous[key]
        write(directory / "result.json", result)
        names = ["identity.json", "best.pt", "last.pt", "val-predictions.json", "val-cases.json", "result.json"]
        if result["wandb"] is not None:
            names.append("wandb-run.json")
        write(directory / "completed-receipt.json", {"files": {n: sha256(directory / n) for n in names}})
        successful = True
        return result
    finally:
        if tracker is not None:
            tracker.finish(exit_code=0 if successful else 1)
