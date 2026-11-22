"""Frozen CLIP text features for training and checkpoint inference."""

import hashlib
from pathlib import Path

import torch
from common.io import sha256
from task2_graph_editing.magicbrush.examples import checked_json
from task2_graph_editing.magicbrush.sequence_model.text import TextStore, load_store, text_key
from task2_graph_editing.magicbrush.structured_model.cache import input_texts
from task2_graph_editing.magicbrush.structured_model.vocabulary import prototype_inputs


def text_store(config, manifest, examples, vocabulary):
    cache = Path(config["text_cache"])
    metadata = checked_json(cache / "manifest.json", manifest["cache_manifest_sha256"])
    store, _ = load_store(cache, metadata["identity"])
    texts = input_texts(examples + prototype_inputs(vocabulary))
    missing = [t for t in texts if text_key(t) not in store.keys]
    if not missing:
        return store, {}
    if sha256(config["weights"]) != config["weights_sha256"]:
        raise ValueError("Frozen CLIP weights changed")
    import clip

    model, _ = clip.load(config["weights"], device="cpu", jit=False)
    model.eval().requires_grad_(False)
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(missing), 32):
            tokens = clip.tokenize(missing[start : start + 32], truncate=True)
            vectors.append(torch.nn.functional.normalize(model.encode_text(tokens).float(), dim=-1))
    delta = torch.cat(vectors)
    keys = [text_key(t) for t in missing]
    return TextStore(list(store.keys) + keys, torch.cat([store.vectors, delta])), {
        "text_keys": keys,
        "vectors_sha256": hashlib.sha256(delta.contiguous().numpy().tobytes()).hexdigest(),
    }
