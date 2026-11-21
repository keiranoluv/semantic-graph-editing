"""Geometry + object-semantic GAT for VG150 PredCls.

The message-passing graph is label-independent. The relation head receives
only ordered object embeddings, never ground-truth predicate labels.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv


class GATPredicateClassifier(nn.Module):
    """Predict predicate logits for ordered pairs after GAT message passing."""

    def __init__(
        self,
        geometry_dim: int = 5,
        num_object_classes: int = 151,
        embedding_dim: int = 32,
        hidden_dim: int = 256,
        num_classes: int = 51,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if min(geometry_dim, num_object_classes, embedding_dim, hidden_dim, num_classes, num_heads, num_layers) <= 0:
            raise ValueError("Model dimensions must be positive")
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in [0, 1)")

        self.geometry_dim = geometry_dim
        self.dropout = dropout
        self.object_embedding = nn.Embedding(num_object_classes, embedding_dim)
        self.node_projection = nn.Linear(geometry_dim + embedding_dim, hidden_dim)
        self.convs = nn.ModuleList(
            [
                GATConv(
                    hidden_dim,
                    hidden_dim // num_heads,
                    heads=num_heads,
                    concat=True,
                    dropout=dropout,
                    add_self_loops=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(num_layers))
        self.predicate_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_label_index: torch.Tensor,
        object_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Return [candidate_pairs, predicates + background] logits.

        edge_index: topology for message passing, built WITHOUT relation labels.
        edge_label_index: pairs to score (can be sampled during training).
        """
        h = self.encode_nodes(x, edge_index, object_labels)
        return self.score_pairs(h, edge_label_index)

    def encode_nodes(self, x: torch.Tensor, edge_index: torch.Tensor, object_labels: torch.Tensor) -> torch.Tensor:
        """Encode objects once so KG fusion can inspect the visual graph context."""
        if x.ndim != 2 or x.shape[1] != self.geometry_dim:
            raise ValueError("Expected geometry x with shape [N, geometry_dim]")
        if object_labels.ndim != 1 or object_labels.shape[0] != x.shape[0]:
            raise ValueError("object_labels must have shape [N]")
        if object_labels.dtype != torch.long:
            raise ValueError("object_labels must be torch.long")
        if object_labels.numel() and (
            bool((object_labels < 0).any()) or bool((object_labels >= self.object_embedding.num_embeddings).any())
        ):
            raise ValueError("object_labels outside vocabulary")
        self._check_edges(edge_index, x.shape[0], "edge_index")

        features = torch.cat((x, self.object_embedding(object_labels)), dim=-1)
        h = F.relu(self.node_projection(features))
        for conv, norm in zip(self.convs, self.norms, strict=True):
            updated = conv(h, edge_index)
            h = norm(h + updated)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        return h

    @staticmethod
    def _check_edges(indices: torch.Tensor, num_nodes: int, name: str) -> None:
        if indices.ndim != 2 or indices.shape[0] != 2 or indices.dtype != torch.long:
            raise ValueError(f"{name} must be torch.long [2, E]")
        if indices.numel() and (bool((indices < 0).any()) or bool((indices >= num_nodes).any())):
            raise ValueError(f"{name} contains invalid node index")

    def score_pairs(self, h: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        """Apply the original predicate head to encoded ordered node pairs."""
        self._check_edges(edge_label_index, h.shape[0], "edge_label_index")
        source, target = edge_label_index
        pair_features = torch.cat((h[source], h[target]), dim=-1)
        return self.predicate_head(pair_features)


def build_gat_model(config: dict) -> GATPredicateClassifier:
    """Construct precisely the model described by a saved run config."""
    if config["model"] != "gat":
        raise ValueError("Checkpoint is not a GAT model")
    return GATPredicateClassifier(
        geometry_dim=config["geometry_dim"],
        num_object_classes=config["num_object_classes"],
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        num_classes=config["num_classes"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    )
