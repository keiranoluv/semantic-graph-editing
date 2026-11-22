"""Full, label-free context topology and separately sampled supervision."""

import torch


def validate_pairs(pairs, nodes, *, full=False):
    if pairs.dtype != torch.long or pairs.ndim != 2 or pairs.shape[0] != 2:
        raise ValueError("Expected long [2,E] pairs")
    if pairs.numel() and ((pairs < 0).any() or (pairs >= nodes).any() or (pairs[0] == pairs[1]).any()):
        raise ValueError("Invalid local pair endpoints/self-loop")
    if full and (
        pairs.shape[1] != nodes * (nodes - 1) or len(torch.unique(pairs[0] * nodes + pairs[1])) != pairs.shape[1]
    ):
        raise ValueError("Context must contain every ordered non-self pair exactly once")
