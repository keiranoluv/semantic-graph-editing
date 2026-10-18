"""Non-graph MLP baseline for predicate classification."""

import torch
from torch import nn


class MLPPredicateClassifier(nn.Module):
    """Predict predicates from ordered subject-object feature pairs."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 51,
    ) -> None:
        super().__init__()

        if min(input_dim, hidden_dim, num_classes) <= 0:
            raise ValueError("Model dimensions must be positive")

        self.input_dim = input_dim
        self.num_classes = num_classes

        self.mlp = nn.Sequential(
            nn.Linear(2 * input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_label_index: torch.Tensor,
    ) -> torch.Tensor:
        """Return logits for ordered candidate pairs.

        Args:
            x: Node features with shape [N, D].
            edge_label_index: Candidate indices with shape [2, E].

        Returns:
            Predicate logits with shape [E, C].
        """
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError("Invalid node feature shape")

        if edge_label_index.ndim != 2 or edge_label_index.shape[0] != 2:
            raise ValueError("Expected edge_label_index shape [2, E]")

        if edge_label_index.dtype != torch.long:
            raise ValueError("Candidate indices must be torch.long")

        src, dst = edge_label_index
        pair_features = torch.cat([x[src], x[dst]], dim=-1)

        return self.mlp(pair_features)
