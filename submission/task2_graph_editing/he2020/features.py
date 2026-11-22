import io
import json
from pathlib import Path

import torch
from common.editing.text import TextStore, text_key
from common.io import fixed, read, sha256, write
from torch import nn

WEIGHTS_SHA256 = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"


def frozen_features(inputs, directory, weights, batch_size=128):
    """CPU-only CLIP text inference; cache contains SOURCE/instruction, no target."""
    from common.editing.data import feature_texts

    directory, weights = Path(directory), Path(weights)
    texts = set()
    for item in inputs:
        row = feature_texts(**item)
        texts.add(row["instruction"])
        texts.update(row["nodes"])
        texts.update(row["edges"])
    texts = sorted(texts)
    identity = {
        "texts_sha256": text_key(json.dumps(texts, ensure_ascii=False)),
        "weights_sha256": WEIGHTS_SHA256,
        "device": "cpu",
        "batch_size": batch_size,
    }
    manifest = directory / "manifest.json"
    if manifest.exists():
        meta = read(manifest)
        if meta["identity"] != identity or meta["vectors_sha256"] != sha256(directory / "vectors.pt"):
            raise ValueError("text_feature_cache_changed")
        return TextStore(meta["text_keys"], torch.load(directory / "vectors.pt", map_location="cpu", weights_only=True))
    if sha256(weights) != WEIGHTS_SHA256:
        raise ValueError("frozen_CLIP_weights_changed")
    import clip
    from clip.simple_tokenizer import SimpleTokenizer

    model, _ = clip.load(str(weights), device="cpu", jit=False)
    model.eval().requires_grad_(False)
    vectors = []
    tokenizer = SimpleTokenizer()
    truncated = sum(len(tokenizer.encode(t)) + 2 > 77 for t in texts)
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            output = model.encode_text(clip.tokenize(texts[start : start + batch_size], truncate=True)).float()
            vectors.append(nn.functional.normalize(output, dim=-1))
            if start % (batch_size * 20) == 0:
                print(
                    json.dumps(
                        {"stage": "CLIP_CPU", "encoded": min(start + batch_size, len(texts)), "total": len(texts)}
                    ),
                    flush=True,
                )
    matrix = torch.cat(vectors)
    buffer = io.BytesIO()
    torch.save(matrix, buffer)
    fixed(directory / "vectors.pt", buffer.getvalue())
    write(
        manifest,
        {
            "identity": identity,
            "text_keys": [text_key(t) for t in texts],
            "shape": list(matrix.shape),
            "vectors_sha256": sha256(directory / "vectors.pt"),
            "truncated_texts": truncated,
            "parameters_frozen": True,
            "target_text_encoded": False,
        },
    )
    return TextStore([text_key(t) for t in texts], matrix)
