"""Frozen WordNet features and edge-removal ablation, separate from baseline."""

import torch

from src.models.gat import GATPredicateClassifier


class KnowledgeGAT(GATPredicateClassifier):
    """Append KG vector + valid bit before the existing GAT projection.

    The no-knowledge control keeps the same architecture but zeros both KG
    inputs. Removing edges retains GAT self loops and all supervised pairs.
    """

    def __init__(self, config, features, valid):
        if features.shape != (config["num_object_classes"], config["kg_dim"]) or valid.shape != features.shape[:1]:
            raise ValueError("KG feature dimensions do not match model config")
        if config["message_edges"] not in {"full", "none"}:
            raise ValueError("Unsupported message edge ablation")
        super().__init__(
            geometry_dim=5 + config["kg_dim"] + 1,
            num_object_classes=config["num_object_classes"],
            embedding_dim=config["embedding_dim"],
            hidden_dim=config["hidden_dim"],
            num_classes=config["num_classes"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
        )
        self.use_knowledge = config["use_knowledge"]
        self.message_edges = config["message_edges"]
        self.register_buffer("kg_features", features.detach().clone())
        self.register_buffer("kg_valid", valid.detach().clone())

    def forward(self, x, edge_index, edge_label_index, object_labels):
        if x.ndim != 2 or x.shape[1] != 5:
            raise ValueError("Expected 5D geometry input")
        if object_labels.dtype != torch.long or object_labels.shape != (x.shape[0],):
            raise ValueError("Invalid object_labels")
        if object_labels.numel() and (object_labels.min() < 0 or object_labels.max() >= self.kg_features.shape[0]):
            raise ValueError("Object label outside KG vocabulary")
        knowledge = self.kg_features[object_labels]
        mask = self.kg_valid[object_labels].to(x.dtype).unsqueeze(-1)
        if not self.use_knowledge:
            knowledge = torch.zeros_like(knowledge)
            mask = torch.zeros_like(mask)
        if self.message_edges == "none":
            edge_index = edge_index[:, :0]
        return super().forward(torch.cat((x, knowledge, mask), dim=-1), edge_index, edge_label_index, object_labels)


def model_from_checkpoint(checkpoint):
    """Restore without needing the original feature folder on another machine."""
    config = checkpoint["config"]
    if checkpoint.get("checkpoint_version") != 4 or config.get("model") != "gat_kg":
        raise ValueError("Expected version 4 GAT+KG checkpoint")
    state = checkpoint["model"]
    model = KnowledgeGAT(config, state["kg_features"], state["kg_valid"])
    model.load_state_dict(state)
    return model
