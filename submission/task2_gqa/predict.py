"""Load one GQA checkpoint for prediction or validation."""

from pathlib import Path

import torch
from common.knowledge.transe.core import sha256
from common.training.gqa_full import create_model, load_vectors

MANIFEST = Path("data/processed/level2/gqa-v1/manifest.json")


def build(job):
    if sha256(Path(job["checkpoint"])) != job["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed after preflight")
    payload = torch.load(job["checkpoint"], map_location="cpu", weights_only=False)
    dims = payload["dimensions"]
    config = {"manifest": str(MANIFEST), "glove": "data/processed/level2/gqa-glove-v1"}
    vectors = load_vectors(config, dims)
    kind, backend = job["model"], job["backend"]
    if backend == "kg":
        from task2_gqa.knowledge import create_model as factory
        from task2_gqa.knowledge_features import load_v2

        features, valid, _ = load_v2(Path("data/knowledge/gqa-wordnet-features-v2"), MANIFEST)
        model = factory(kind, dims, features, valid)
    elif backend in ("fast", "ablation"):
        if backend == "fast":
            from task2_gqa.training.engine import create_model as factory
        else:
            from task2_gqa.models.ablation import create_model as factory
        model = factory(kind, dims, vectors)
    else:
        model = create_model(kind, dims, vectors)
    model.load_state_dict(payload["model"], strict=True)
    return model.eval(), dims
