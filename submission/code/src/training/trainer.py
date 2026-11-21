"""Training utilities for the MLP predicate baseline."""

import torch
from torch import nn
from torch_geometric.data import Data

from src.training.sampling import sample_training_pairs


def train_step(
    model: nn.Module,
    graph: Data,
    optimizer: torch.optim.Optimizer,
    negative_ratio: int = 3,
    generator: torch.Generator | None = None,
) -> float | None:
    """Train on one graph and return the scalar loss.

    Returns None if the graph contains no positive pairs.
    """
    model.train()

    indices = sample_training_pairs(
        graph.edge_label,
        negative_ratio=negative_ratio,
        generator=generator,
    )

    if indices.numel() == 0:
        return None

    candidates = graph.edge_label_index[:, indices]
    targets = graph.edge_label[indices]

    optimizer.zero_grad()

    logits = model(
        x=graph.x,
        edge_label_index=candidates,
    )

    loss = nn.functional.cross_entropy(logits, targets)

    loss.backward()
    optimizer.step()

    return float(loss.detach().item())
