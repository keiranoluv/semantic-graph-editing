"""RotatE ranking must use the same filtered protocol as TransE."""

import numpy as np
import pytest
import torch

from src.knowledge.rotate import Rotate, RotateV2, evaluate_rotate
from src.knowledge.transe.core import negative_sample, set_seed


def test_rotate_filtered_training_and_ranking_match_candidate_reference():
    set_seed(7)
    model = Rotate(19, 2, 4, margin=1.0)
    train = np.array([[0, 0, 1], [3, 1, 4], [7, 0, 9]], dtype=np.int64)
    heldout = np.array([[2, 0, 1]], dtype=np.int64)
    train_known = {tuple(row) for row in train}
    known = train_known | {tuple(heldout[0]), (0, 0, 5), (8, 0, 9)}
    negatives = negative_sample(train, 19, train_known, np.random.default_rng(3))
    assert not set(map(tuple, negatives)) & train_known
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.zero_grad()
    loss = model.filtered_loss(torch.as_tensor(train), torch.as_tensor(negatives))
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()

    observed = evaluate_rotate(model, train, known, torch.device("cpu"))
    ranks = []
    with torch.no_grad():
        for h, r, t in train:
            for side in (0, 2):
                query = np.tile([h, r, t], (19, 1))
                query[:, side] = np.arange(19)
                scores = model.score(torch.as_tensor(query, dtype=torch.long)).tolist()
                gold_id = h if side == 0 else t
                gold = scores[gold_id]
                for candidate in range(19):
                    triple = (candidate, int(r), int(t)) if side == 0 else (int(h), int(r), candidate)
                    if triple in known and triple != (int(h), int(r), int(t)):
                        scores[candidate] = float("inf")
                better = sum(score < gold - 1e-8 for score in scores)
                tied = sum(abs(score - gold) <= 1e-8 for score in scores)
                ranks.append(1 + better + (tied - 1) / 2)
    assert observed["count"] == len(ranks)
    assert observed["MRR"] == pytest.approx(np.mean([1 / rank for rank in ranks]))
    for k in (1, 3, 10):
        assert observed[f"Hits@{k}"] == pytest.approx(np.mean([rank <= k for rank in ranks]))


def test_rotate_v2_self_adversarial_loss_and_filtered_ranking():
    set_seed(9)
    model = RotateV2(17, 2, 4, margin=6.0)
    positives = np.array([[0, 0, 1], [3, 1, 4]], dtype=np.int64)
    known = {(0, 0, 1), (3, 1, 4), (2, 0, 1), (0, 0, 5)}
    rng = np.random.default_rng(11)
    negatives = np.stack([negative_sample(positives, 17, known, rng) for _ in range(8)], axis=1)
    assert all(tuple(triple) not in known for triple in negatives.reshape(-1, 3))
    loss = model.self_adversarial_loss(torch.as_tensor(positives), torch.as_tensor(negatives), 1.0)
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    assert model.node_emb.weight.grad is not None and torch.isfinite(model.node_emb.weight.grad).all()

    observed = evaluate_rotate(model, positives, known, torch.device("cpu"))
    ranks = []
    with torch.no_grad():
        for h, r, t in positives:
            for side in (0, 2):
                query = np.tile([h, r, t], (17, 1))
                query[:, side] = np.arange(17)
                scores = model.score(torch.as_tensor(query, dtype=torch.long)).tolist()
                gold_id = h if side == 0 else t
                gold = scores[gold_id]
                for candidate in range(17):
                    triple = (candidate, int(r), int(t)) if side == 0 else (int(h), int(r), candidate)
                    if triple in known and triple != (int(h), int(r), int(t)):
                        scores[candidate] = float("inf")
                better = sum(score < gold - 1e-8 for score in scores)
                tied = sum(abs(score - gold) <= 1e-8 for score in scores)
                ranks.append(1 + better + (tied - 1) / 2)
    assert observed["MRR"] == pytest.approx(np.mean([1 / rank for rank in ranks]))
