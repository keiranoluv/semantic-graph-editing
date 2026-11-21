"""Separate frozen input-only text caches for development and post-selection test."""
import gc
import os

import torch

from tools.edit_model_v1.data import feature_texts
from tools.edit_model_v1.text import load_store, text_key
from tools.emnlp2020_graph_modification_v1.dataset import sha256, write


def input_texts(examples):
    texts = set()
    for example in examples:
        item = feature_texts(**example.model_input())
        texts.add(item["instruction"])
        texts.update(item["nodes"])
        texts.update(item["edges"])
    return sorted(texts, key=lambda t: (text_key(t), t))


def build_cache(examples, config, identity, output, root, *, scope):
    if scope not in {"trainval", "test_after_selection"}:
        raise ValueError("Unknown text cache scope")
    if (output / "manifest.json").exists():
        return load_store(output, identity)
    # Identity is durable before inference; uncommitted vectors can be rebuilt on resume.
    write(output / "identity.json", identity)
    weights = root / config["weights"]
    if sha256(weights) != config["weights_sha256"]:
        raise ValueError("Pinned CLIP weights changed")
    import clip
    from clip.simple_tokenizer import SimpleTokenizer

    texts = input_texts(examples)
    if not texts:
        raise ValueError("Empty text cache")
    torch.manual_seed(42)
    model, _ = clip.load(str(weights), device="cuda", jit=False)
    model.eval().requires_grad_(False)
    tokenizer = SimpleTokenizer()
    truncated = [text_key(t) for t in texts if len(tokenizer.encode(t)) + 2 > 77]
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(texts), config["text_batch_size"]):
            tokens = clip.tokenize(texts[start : start + config["text_batch_size"]], truncate=True).to("cuda")
            chunks.append(torch.nn.functional.normalize(model.encode_text(tokens).float(), dim=-1).cpu())
    vectors = torch.cat(chunks)
    if vectors.shape != (len(texts), 512) or not torch.isfinite(vectors).all():
        raise ValueError("Invalid CLIP cache")
    temporary = output / "vectors.pt.tmp"
    torch.save(vectors, temporary)
    os.replace(temporary, output / "vectors.pt")
    metadata = {
        "identity": identity,
        "vectors_sha256": sha256(output / "vectors.pt"),
        "shape": list(vectors.shape),
        "text_sha256": [text_key(t) for t in texts],
        "truncated_text_sha256": truncated,
        "parameters_frozen": True,
        "context_length": 77,
        "scope": scope,
        "target_text_encoded": False,
        "input_ids": [e.id for e in examples],
    }
    write(output / "manifest.json", metadata)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return load_store(output, identity)
