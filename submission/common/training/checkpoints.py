"""VG150 checkpoint publication, run directories and RNG restoration."""

import hashlib
import os
import random
import re
from pathlib import Path

import torch


def resolve_device(requested):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(requested)


def create_run_dir(output_dir, run_name):
    """Create a new run atomically; never reuse a previous run's outputs."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", run_name):
        raise ValueError("Run name must be a safe directory name")
    root = output_dir / run_name
    root.mkdir(parents=True, exist_ok=True)
    for number in range(1, 100000):
        path = root / f"run_{number:04d}"
        try:
            path.mkdir(exist_ok=False)
            return path
        except FileExistsError:
            continue
    raise RuntimeError("Too many runs with the same name")


def restore_training_state(checkpoint, model, optimizer, generator, device):
    """Restore all training-critical state at the end of an epoch."""
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    generator.set_state(checkpoint["sampler_rng"])
    torch.set_rng_state(checkpoint["torch_rng"])
    if device.type == "cuda":
        cuda_rng = checkpoint.get("cuda_rng")
        if cuda_rng is None:
            raise ValueError("CUDA RNG state missing from checkpoint")
        torch.cuda.set_rng_state_all(cuda_rng)
    random.setstate(checkpoint["python_rng"])
    return (
        int(checkpoint["epoch"]) + 1,
        float(checkpoint["best_score"]),
        int(checkpoint["best_epoch"]),
        int(checkpoint["bad_epochs"]),
        list(checkpoint["history"]),
    )


def atomic_torch_save(value, path):
    """Replace a completed checkpoint atomically within its own run."""
    temp = path.with_name(path.name + ".tmp")
    try:
        torch.save(value, temp)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def atomic_text_save(content, path):
    temp = path.with_name(path.name + ".tmp")
    try:
        temp.write_text(content, encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


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


def state_hash(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(str((tuple(value.shape), str(value.dtype))).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
