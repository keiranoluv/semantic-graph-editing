"""Checkpoint and RNG helpers shared by graph-edit trainers."""

import os
from pathlib import Path

import torch


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
