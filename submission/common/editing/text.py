"""Frozen CLIP text cache. GPU entry point belongs to the user's launcher only."""

import hashlib
import json
from pathlib import Path

import torch
from common.knowledge.transe import sha256


def text_key(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class TextStore:
    def __init__(self, texts, vectors):
        if vectors.ndim != 2 or len(texts) != len(vectors) or not torch.isfinite(vectors).all():
            raise ValueError("Invalid frozen text matrix")
        if len(set(texts)) != len(texts):
            raise ValueError("Repeated cache text keys")
        self.keys = {text: i for i, text in enumerate(texts)}
        self.vectors = vectors.detach().float().cpu()
        self.dim = vectors.shape[1]

    def __getitem__(self, text):
        if text_key(text) not in self.keys:
            raise ValueError("Missing frozen source/instruction feature; rebuild a versioned cache")
        return self.vectors[self.keys[text_key(text)]]


def load_store(directory, identity):
    directory = Path(directory)
    metadata = json.loads((directory / "manifest.json").read_text())
    if metadata["identity"] != identity or sha256(directory / "vectors.pt") != metadata["vectors_sha256"]:
        raise ValueError("Frozen text cache identity/hash mismatch")
    vectors = torch.load(directory / "vectors.pt", map_location="cpu", weights_only=True)
    if metadata["shape"] != list(vectors.shape):
        raise ValueError("Frozen text shape metadata differs")
    return TextStore(metadata["text_sha256"], vectors), metadata
