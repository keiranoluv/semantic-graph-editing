"""Construct the GQA knowledge-aware classifier used by saved checkpoints."""

from common.models.gat_kg import KnowledgeGAT
from task2_gqa import evaluation as evaluator

KINDS = ("gat", "gat-kg")


def create_model(kind, dimensions, features, valid):
    if kind == "gat":
        return evaluator.build_model(kind, dimensions)
    if kind not in KINDS:
        raise ValueError(f"Unsupported GQA KG model: {kind}")
    settings = {
        **dimensions,
        "kg_dim": features.shape[1],
        "use_knowledge": kind == "gat-kg",
        "message_edges": "full",
        "num_heads": 4,
        "num_layers": 2,
        "dropout": 0.1,
    }
    return KnowledgeGAT(settings, features, valid)
