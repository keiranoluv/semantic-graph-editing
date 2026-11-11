"""Reviewed ConceptNet ComplEx scoring with filtered KG evaluation."""

import math

import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.nn.kge import ComplEx


class ComplexReviewed(ComplEx):
    """PyG ComplEx with dimension-scaled initialization and logistic loss."""

    def reset_parameters(self) -> None:
        std = 1.0 / math.sqrt(self.hidden_channels)
        for embedding in (self.node_emb, self.node_emb_im, self.rel_emb, self.rel_emb_im):
            torch.nn.init.normal_(embedding.weight, mean=0.0, std=std)

    def score(self, triples: torch.Tensor) -> torch.Tensor:
        return self(triples[:, 0], triples[:, 1], triples[:, 2])

    def filtered_loss(self, positive: torch.Tensor, negatives: torch.Tensor, l2_regularization: float) -> torch.Tensor:
        if negatives.ndim != 3 or negatives.shape[0] != positive.shape[0] or negatives.shape[2] != 3:
            raise ValueError("Expected negatives with shape [batch, count, 3]")
        positive_score = self.score(positive)
        negative_score = self.score(negatives.reshape(-1, 3)).reshape(negatives.shape[:2])
        logistic = (F.softplus(-positive_score).mean() + F.softplus(negative_score).mean()) / 2
        entity_norm = (self.node_emb.weight.square() + self.node_emb_im.weight.square()).sum(dim=1).mean()
        relation_norm = (self.rel_emb.weight.square() + self.rel_emb_im.weight.square()).sum(dim=1).mean()
        return logistic + l2_regularization * (entity_norm + relation_norm)


def evaluate_complex(
    model: ComplexReviewed, triples: np.ndarray, known: set[tuple[int, int, int]], device: torch.device
) -> dict:
    """Filtered head/tail ranking with the same tie policy as TransE/RotatE."""
    head_known: dict[tuple[int, int], set[int]] = {}
    tail_known: dict[tuple[int, int], set[int]] = {}
    for h, r, t in known:
        head_known.setdefault((int(r), int(t)), set()).add(int(h))
        tail_known.setdefault((int(h), int(r)), set()).add(int(t))
    ranks = []
    model.eval()
    with torch.no_grad():
        entity_re, entity_im = model.node_emb.weight, model.node_emb_im.weight
        relation_re, relation_im = model.rel_emb.weight, model.rel_emb_im.weight
        for h, r, t in triples:
            h, r, t = int(h), int(r), int(t)
            for side in (0, 2):
                if side == 0:
                    coefficient_re = relation_re[r] * entity_re[t] + relation_im[r] * entity_im[t]
                    coefficient_im = relation_re[r] * entity_im[t] - relation_im[r] * entity_re[t]
                    gold_id = h
                    filtered = head_known.get((r, t), ())
                else:
                    coefficient_re = entity_re[h] * relation_re[r] - entity_im[h] * relation_im[r]
                    coefficient_im = entity_im[h] * relation_re[r] + entity_re[h] * relation_im[r]
                    gold_id = t
                    filtered = tail_known.get((h, r), ())
                # Explicit products avoid CUDA matvec/CuBLAS, which requires an
                # extra workspace setting when deterministic algorithms are on.
                scores = (entity_re * coefficient_re).sum(dim=1) + (entity_im * coefficient_im).sum(dim=1)
                gold = scores[gold_id]
                excluded = [index for index in filtered if index != gold_id]
                if excluded:
                    scores[torch.as_tensor(excluded, dtype=torch.long, device=device)] = -float("inf")
                better = int(torch.count_nonzero(scores > gold + 1e-8).item())
                tied = int(torch.count_nonzero(torch.abs(scores - gold) <= 1e-8).item())
                ranks.append(1 + better + (tied - 1) / 2)
    return {
        "MRR": float(np.mean([1 / x for x in ranks])),
        "Hits@1": float(np.mean([x <= 1 for x in ranks])),
        "Hits@3": float(np.mean([x <= 3 for x in ranks])),
        "Hits@10": float(np.mean([x <= 10 for x in ranks])),
        "count": len(ranks),
    }
