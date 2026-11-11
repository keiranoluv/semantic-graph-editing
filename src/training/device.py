"""Torch execution-device selection."""

import torch


def resolve_device(name: str) -> torch.device:
    if name not in ("auto", "cpu", "cuda"):
        raise ValueError(f"Unsupported device: {name}")

    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"

    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    return torch.device(name)
