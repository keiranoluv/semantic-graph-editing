"""Code-reference PE-Net PL/PR with explicit multi-positive targets and dynamic C.

Loss behavior adapted from VL-Group/PENET@9c9f50777c66647799cb7a17dfb855aa98aecd1e.
Copyright (c) 2018 Facebook. MIT notice: licenses/PENET-MIT.txt.
"""

import torch
import torch.nn.functional as F


def squared_distances(a, b):
    """O(E*C) storage instead of [E,C,D]; clamp only roundoff-negative distances."""
    return (a.square().sum(-1, keepdim=True) + b.square().sum(-1).unsqueeze(0) - 2 * (a @ b.T)).clamp_min(0)


def prototype_losses(logits, relation, prototypes, targets, *, mode="plpr", k1=10, k2=1, gamma1=1.0, gamma2=7.0):
    if mode not in ("ce", "pl", "plpr") or min(k1, k2) < 1 or min(gamma1, gamma2) < 0:
        raise ValueError("Invalid loss configuration")
    if logits.ndim != 2 or targets.shape != logits.shape or relation.ndim != 2 or prototypes.ndim != 2:
        raise ValueError("Invalid PE-Net loss shapes")
    e, c = logits.shape
    if c < 2 or relation.shape[0] != e or prototypes.shape != (c, relation.shape[1]):
        raise ValueError("Embedding/class dimensions disagree")
    if any(not torch.isfinite(t).all() for t in (logits, relation, prototypes, targets)):
        raise ValueError("Non-finite loss input")
    if (targets < 0).any() or not torch.allclose(targets.sum(1), targets.new_ones(e), atol=1e-6):
        raise ValueError("Targets must be normalized distributions")
    positive = targets > 0
    if ((positive[:, 0]) & positive[:, 1:].any(1)).any() or (positive.sum(1) == c).any():
        raise ValueError("Background cannot coexist with foreground; negative prototypes required")
    zero = logits.sum() * 0
    losses = {name: zero for name in ("ce", "pl_euclidean", "pr_cosine", "pr_euclidean")}
    if e == 0:
        return {**losses, "total": zero}  # Caller must skip optimizer update, including PR.
    losses["ce"] = -(targets * F.log_softmax(logits, dim=1)).sum(1).mean()
    if mode in ("pl", "plpr"):
        distances = squared_distances(relation, prototypes)
        nearest = distances.masked_fill(positive, float("inf")).topk(min(k1, c), largest=False, dim=1).values
        valid = torch.isfinite(nearest)
        negative = nearest.masked_fill(~valid, 0).sum(1) / valid.sum(1).clamp_min(1)
        losses["pl_euclidean"] = (F.relu(distances - negative[:, None] + gamma1) * targets).sum(1).mean()
    if mode == "plpr":
        unit = F.normalize(prototypes, dim=1, eps=1e-8)
        similarity = unit @ unit.detach().T
        losses["pr_cosine"] = similarity.norm(p=2, dim=1).sum() / (c * c)
        distances = squared_distances(prototypes, prototypes.detach())
        self_mask = torch.eye(c, device=prototypes.device, dtype=torch.bool)
        nearest = (
            distances.masked_fill(self_mask, float("inf")).topk(min(k2, c - 1), largest=False, dim=1).values.mean(1)
        )
        # Mean of per-prototype hinges, following CODE (not hinge of a global mean).
        losses["pr_euclidean"] = F.relu(gamma2 - nearest).mean()
    return {**losses, "total": sum(losses.values())}
