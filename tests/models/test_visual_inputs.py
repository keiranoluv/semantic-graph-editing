"""All comparison architectures accept the same cached CLIP node width."""

import torch

from src.models.gat import GATPredicateClassifier
from src.models.gat_kg import KnowledgeGAT
from src.models.mlp_semantic import SemanticMLPPredicateClassifier


def test_matched_visual_input_width():
    x = torch.randn(3, 517)
    labels = torch.tensor([1, 2, 3])
    pairs = torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])
    mlp = SemanticMLPPredicateClassifier(geometry_dim=517)
    gat = GATPredicateClassifier(geometry_dim=517)
    kg_config = {
        "geometry_dim": 517,
        "num_object_classes": 151,
        "kg_dim": 64,
        "embedding_dim": 32,
        "hidden_dim": 256,
        "num_classes": 51,
        "num_heads": 4,
        "num_layers": 2,
        "dropout": 0.1,
        "use_knowledge": True,
        "message_edges": "full",
    }
    kg = KnowledgeGAT(kg_config, torch.randn(151, 64), torch.ones(151, dtype=torch.bool))
    assert mlp(x, pairs, labels).shape == (6, 51)
    assert gat(x, pairs, pairs, labels).shape == (6, 51)
    assert kg(x, pairs, pairs, labels).shape == (6, 51)
