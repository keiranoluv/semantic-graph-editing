"""Construct the reported AlignG ablation models."""

from task2_gqa.models.aligng import AlignGConfig
from task2_gqa.models.fast import FastAlignG, FastConfig

VARIANTS = {
    "edge-only": {
        "feedback_enabled": True,
        "prototype_update_enabled": False,
        "reverse_feedback_enabled": True,
        "loss_mode": "plpr",
    },
    "pe-ce-aligng": {
        "feedback_enabled": True,
        "prototype_update_enabled": True,
        "reverse_feedback_enabled": True,
        "loss_mode": "ce",
    },
}

MODELS = tuple(VARIANTS)


def create_model(kind, dimensions, vectors):
    if kind not in MODELS:
        raise ValueError("Unknown AlignG variant")
    return FastAlignG(
        *vectors,
        semantic_dim=dimensions["semantic_dim"],
        fast_config=FastConfig(**dimensions["fast"]),
        dropout=dimensions["pe_dropout"],
        config=AlignGConfig(**{**dimensions["aligng"], **VARIANTS[kind]}),
    )
