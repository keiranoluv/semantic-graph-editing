"""Memory-bounded pair head, exact gradient accumulation and diagnostics."""

import contextlib
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


def autocast(device, precision):
    if torch.device(device).type == "cuda" and precision == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def denominators(batch):
    counts = torch.zeros(batch["sample_count"], device=batch["query_batch"].device)
    counts.index_add_(0, batch["query_batch"], batch["supervision_mask"].sum(dim=1).float())
    return counts.clamp_min(1)


def loss_chunk(logits, batch, low, high, normalizers):
    labels, valid = batch["labels"][low:high], batch["supervision_mask"][low:high]
    losses = F.binary_cross_entropy_with_logits(logits.float(), labels, reduction="none")
    per_query = (losses * valid).sum(dim=1) / normalizers[batch["query_batch"][low:high]]
    return per_query.sum(), losses.detach(), labels, valid


def backward_microbatch(model, batch, *, chunk_size, group_windows, precision="fp32"):
    """Accumulate head gradients by chunks, then backpropagate encoder ONCE.

    The leaf proxy prevents each chunk from retaining its head activations or
    re-running GNN backward. Normalization equals the mean window BCE over the
    actual optimizer group, including a final partial group.
    """
    device = batch["visual"].device
    normalizers = denominators(batch)
    with autocast(device, precision):
        encoded = model.encode(batch)
    proxy = encoded.detach().requires_grad_(True)
    q = batch["query_edges"].shape[1]
    totals = torch.zeros(9, device=device, dtype=torch.float64)
    for low in range(0, q, chunk_size):
        high = min(q, low + chunk_size)
        with autocast(device, precision):
            logits = model.score(proxy, batch, low, high)
        value, losses, labels, valid = loss_chunk(logits, batch, low, high, normalizers)
        (value / group_windows).backward()
        positive = (labels > 0) & valid
        negative = (labels == 0) & valid
        predicted = (logits.detach() >= 0) & valid
        totals += torch.stack(
            (
                value.detach().double(),
                (losses * positive).sum().double(),
                positive.sum().double(),
                (losses * negative).sum().double(),
                negative.sum().double(),
                (predicted & positive).sum().double(),
                (predicted & negative).sum().double(),
                (~predicted & positive).sum().double(),
                predicted.sum().double(),
            )
        )
    if proxy.grad is not None:
        encoded.backward(proxy.grad)
    if not torch.isfinite(totals).all():
        raise ValueError("Non-finite training loss")
    return totals.cpu().numpy()


def diagnostics(totals, windows):
    loss, pos_loss, positives, neg_loss, negatives, tp, fp, fn, active = map(float, totals)
    return {
        "loss": loss / max(windows, 1),
        "positive_bce": pos_loss / max(positives, 1),
        "negative_bce": neg_loss / max(negatives, 1),
        "positive_labels": int(positives),
        "frame_precision": tp / max(tp + fp, 1),
        "frame_recall": tp / max(tp + fn, 1),
        "frame_f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "active_fraction": active / max(positives + negatives, 1),
    }


@torch.no_grad()
def predict_batch(model, batch, chunk_size, precision="fp32"):
    with autocast(batch["visual"].device, precision):
        h = model.encode(batch)
        parts = [
            model.score(h, batch, low, low + chunk_size).float().cpu().numpy()
            for low in range(0, batch["query_edges"].shape[1], chunk_size)
        ]
    return np.concatenate(parts) if parts else np.empty((0, 57), dtype=np.float32)


def selection_key(report):
    primary = report["metrics"]["0.5"]["50"]
    if primary["mR"] is None or primary["R"] is None:
        raise ValueError("Validation has no GT support")
    return float(primary["mR"]), float(primary["R"])


def should_stop(epoch, bad_epochs, cfg):
    return epoch >= cfg["min_epochs"] and bad_epochs >= cfg["early_stopping_patience"]


def save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial.pt")
    torch.save(payload, tmp)
    tmp.replace(path)


def restore_checkpoint(path, model, optimizer, scheduler, expected_signature, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["signature"] != expected_signature:
        raise ValueError("Checkpoint config/code/features/profile differ from current run")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor) and key != "step":
                state[key] = value.to(device)
    torch.set_rng_state(checkpoint["torch_rng"])
    if torch.device(device).type == "cuda":
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    return checkpoint
