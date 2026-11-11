"""ComplEx score and filtered ranking against direct candidate enumeration."""

import numpy as np
import pytest
import torch

from src.knowledge.complex import ComplexReviewed, evaluate_complex
from src.knowledge.transe.core import negative_sample, set_seed


def test_complex_filtered_loss_and_ranking_match_reference():
    set_seed(17)
    model = ComplexReviewed(19, 2, 4)
    positives = np.array([[0, 0, 1], [3, 1, 4], [7, 0, 9]], dtype=np.int64)
    known = {(0, 0, 1), (2, 0, 1), (0, 0, 5), (3, 1, 4), (7, 0, 9), (8, 0, 9)}
    rng = np.random.default_rng(12)
    negatives = np.stack([negative_sample(positives, 19, known, rng) for _ in range(4)], axis=1)
    assert all(tuple(triple) not in known for triple in negatives.reshape(-1, 3))
    loss = model.filtered_loss(torch.as_tensor(positives), torch.as_tensor(negatives), 0.001)
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    assert model.node_emb.weight.grad is not None and torch.isfinite(model.node_emb.weight.grad).all()

    observed = evaluate_complex(model, positives, known, torch.device("cpu"))
    ranks = []
    with torch.no_grad():
        for h, r, t in positives:
            for side in (0, 2):
                candidates = np.tile([h, r, t], (19, 1))
                candidates[:, side] = np.arange(19)
                scores = model.score(torch.as_tensor(candidates, dtype=torch.long)).tolist()
                gold_id = h if side == 0 else t
                gold = scores[gold_id]
                for candidate in range(19):
                    triple = (candidate, int(r), int(t)) if side == 0 else (int(h), int(r), candidate)
                    if triple in known and triple != (int(h), int(r), int(t)):
                        scores[candidate] = -float("inf")
                better = sum(score > gold + 1e-8 for score in scores)
                tied = sum(abs(score - gold) <= 1e-8 for score in scores)
                ranks.append(1 + better + (tied - 1) / 2)
    assert observed["count"] == len(ranks)
    assert observed["MRR"] == pytest.approx(np.mean([1 / rank for rank in ranks]))
    for k in (1, 3, 10):
        assert observed[f"Hits@{k}"] == pytest.approx(np.mean([rank <= k for rank in ranks]))
