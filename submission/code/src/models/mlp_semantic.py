"""MLP baseline with trainable object-label embeddings."""

import torch
from torch import nn


class SemanticMLPPredicateClassifier(nn.Module):
    """Predict predicates from geometry and object class embeddings."""

    def __init__(
        self,
        geometry_dim: int = 5,
        num_object_classes: int = 151,
        embedding_dim: int = 32,
        hidden_dim: int = 256,
        num_classes: int = 51,
    ) -> None:
        super().__init__()

        if (
            min(
                geometry_dim,
                num_object_classes,
                embedding_dim,
                hidden_dim,
                num_classes,
            )
            <= 0
        ):
            raise ValueError("Model dimensions must be positive")

        self.geometry_dim = geometry_dim
        self.num_classes = num_classes

        self.object_embedding = nn.Embedding(
            num_embeddings=num_object_classes,
            embedding_dim=embedding_dim,
        )

        node_dim = geometry_dim + embedding_dim

        self.mlp = nn.Sequential(
            nn.Linear(2 * node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_label_index: torch.Tensor,
        object_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Return predicate logits [E, C]."""
        if x.ndim != 2 or x.shape[1] != self.geometry_dim:
            raise ValueError("Invalid geometry feature shape")

        if object_labels.shape != (x.shape[0],):
            raise ValueError("Invalid object label shape")

        if object_labels.dtype != torch.long:
            raise ValueError("Object labels must be torch.long")

        if edge_label_index.ndim != 2 or edge_label_index.shape[0] != 2:
            raise ValueError("Expected edge_label_index [2, E]")

        if edge_label_index.dtype != torch.long:
            raise ValueError("Candidate indices must be torch.long")

        if object_labels.numel():
            if object_labels.min() < 0:
                raise ValueError("Negative object label")
            if object_labels.max() >= self.object_embedding.num_embeddings:
                raise ValueError("Object label exceeds vocabulary")

        semantic = self.object_embedding(object_labels)
        node_features = torch.cat((x, semantic), dim=-1)

        subject, obj = edge_label_index

        pair_features = torch.cat(
            (node_features[subject], node_features[obj]),
            dim=-1,
        )

        return self.mlp(pair_features)
