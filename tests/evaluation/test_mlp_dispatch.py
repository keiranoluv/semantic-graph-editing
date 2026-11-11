"""Test evaluation dispatch for MLP baselines."""

import pytest
import torch
from torch_geometric.data import Data

from scripts.evaluate_mlp import build_model, predict
from src.models.mlp import MLPPredicateClassifier
from src.models.mlp_semantic import SemanticMLPPredicateClassifier


def test_geometry_model_dispatch():
    model = build_model(
        {
            "model": "mlp",
            "hidden_dim": 256,
            "num_classes": 51,
        }
    )

    assert isinstance(model, MLPPredicateClassifier)


def test_semantic_model_dispatch():
    model = build_model(
        {
            "model": "mlp_semantic",
            "hidden_dim": 256,
            "embedding_dim": 32,
            "num_object_classes": 151,
            "num_classes": 51,
        }
    )

    assert isinstance(model, SemanticMLPPredicateClassifier)


def test_semantic_prediction():
    config = {
        "model": "mlp_semantic",
        "hidden_dim": 32,
        "embedding_dim": 8,
        "num_object_classes": 10,
        "num_classes": 3,
    }

    model = build_model(config)

    graph = Data(
        x=torch.randn(3, 5),
        object_labels=torch.tensor([1, 2, 3]),
        edge_label_index=torch.tensor([[0, 1], [1, 2]]),
    )

    logits = predict(model, graph, config["model"])

    assert logits.shape == (2, 3)
    assert torch.isfinite(logits).all()


def test_unknown_model_rejected():
    with pytest.raises(ValueError, match="Unsupported model"):
        build_model({"model": "unknown"})
