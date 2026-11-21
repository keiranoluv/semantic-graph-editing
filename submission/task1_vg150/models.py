"""Reconstruct VG150 MLP architectures from checkpoint configurations."""

from common.models.mlp import MLPPredicateClassifier
from common.models.mlp_semantic import SemanticMLPPredicateClassifier


def build_model(config):
    """Reconstruct the model architecture from checkpoint configuration."""
    model_name = config["model"]

    if model_name in ("mlp", "mlp_geometry"):
        return MLPPredicateClassifier(
            input_dim=config.get("geometry_dim", 5),
            hidden_dim=config["hidden_dim"],
            num_classes=config["num_classes"],
        )

    if model_name == "mlp_semantic":
        return SemanticMLPPredicateClassifier(
            geometry_dim=config.get("geometry_dim", 5),
            num_object_classes=config["num_object_classes"],
            embedding_dim=config["embedding_dim"],
            hidden_dim=config["hidden_dim"],
            num_classes=config["num_classes"],
        )

    raise ValueError(f"Unsupported model: {model_name}")
