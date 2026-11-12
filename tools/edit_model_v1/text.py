"""Frozen CLIP text cache. GPU entry point belongs to the user's launcher only."""

import hashlib
import io
import json
from pathlib import Path

import torch

from src.knowledge.transe.core import sha256
from tools.edit_model_v1.data import feature_texts
from tools.prepare_magicbrush import ROOT, encode, fixed


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


def collect_texts(examples):
    texts = set()
    for split in ("train", "val"):
        for example in examples[split]:
            item = feature_texts(**example.model_input())
            texts.add(item["instruction"])
            texts.update(item["nodes"])
            texts.update(item["edges"])
    return sorted(texts, key=lambda text: (text_key(text), text))


def load_store(directory, identity):
    directory = Path(directory)
    metadata = json.loads((directory / "manifest.json").read_text())
    if metadata["identity"] != identity or sha256(directory / "vectors.pt") != metadata["vectors_sha256"]:
        raise ValueError("Frozen text cache identity/hash mismatch")
    vectors = torch.load(directory / "vectors.pt", map_location="cpu", weights_only=True)
    if metadata["shape"] != list(vectors.shape):
        raise ValueError("Frozen text shape metadata differs")
    return TextStore(metadata["text_sha256"], vectors), metadata


def cache_texts(examples, config, identity, output):
    output = Path(output)
    if (output / "manifest.json").exists():
        return load_store(output, identity)[1]
    import clip
    from clip.simple_tokenizer import SimpleTokenizer

    weights = ROOT / config["weights"]
    if sha256(weights) != config["weights_sha256"]:
        raise ValueError("Frozen CLIP weights changed")
    texts = collect_texts(examples)
    if not texts:
        raise ValueError("No eligible source/instruction texts")
    torch.manual_seed(42)
    model, _ = clip.load(str(weights), device="cuda", jit=False)
    model.eval().requires_grad_(False)
    tokenizer = SimpleTokenizer()
    truncated = [text_key(t) for t in texts if len(tokenizer.encode(t)) + 2 > 77]
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(texts), config["text_batch_size"]):
            tokens = clip.tokenize(texts[start : start + config["text_batch_size"]], truncate=True).to("cuda")
            features = model.encode_text(tokens).float()
            vectors.append(torch.nn.functional.normalize(features, dim=-1).cpu())
    matrix = torch.cat(vectors)
    if matrix.shape[1] != 512 or not torch.isfinite(matrix).all():
        raise ValueError("Unexpected CLIP text features")
    stream = io.BytesIO()
    torch.save(matrix, stream)
    fixed(output / "vectors.pt", stream.getvalue())
    metadata = {
        "identity": identity,
        "vectors_sha256": sha256(output / "vectors.pt"),
        "text_sha256": [text_key(t) for t in texts],
        "shape": list(matrix.shape),
        "truncated_text_sha256": truncated,
        "context_length": 77,
        "parameters_frozen": True,
        "model_inputs": "source node label+attributes, source predicates, instruction only",
        "target_text_encoded": False,
        "test_used": False,
    }
    fixed(output / "manifest.json", encode(metadata))
    return metadata
