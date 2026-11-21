"""Training-only candidate-pair sampling."""

import torch


def sample_training_pairs(
    edge_labels: torch.Tensor,
    negative_ratio: int = 3,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Select positives and randomly sampled background pairs.

    Label convention:
        -1: ambiguous, ignored
         0: background
        >0: positive predicate

    Returns:
        Indices into the original candidate-pair dimension.
    """
    if edge_labels.ndim != 1:
        raise ValueError("edge_labels must be one-dimensional")

    if negative_ratio < 0:
        raise ValueError("negative_ratio must be nonnegative")

    positive = torch.nonzero(edge_labels > 0).flatten()
    negative = torch.nonzero(edge_labels == 0).flatten()

    if positive.numel() == 0:
        return positive

    num_negative = min(
        negative.numel(),
        negative_ratio * positive.numel(),
    )

    permutation = torch.randperm(
        negative.numel(),
        generator=generator,
        device=negative.device,
    )

    selected_negative = negative[permutation[:num_negative]]

    return torch.cat([positive, selected_negative])
