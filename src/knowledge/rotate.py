"""PyG RotatE with the project's filtered-negative and ranking protocol."""

import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.nn.kge import RotatE


class Rotate(RotatE):
    """Expose distance and a margin loss using caller-supplied filtered negatives."""

    def score(self, triples: torch.Tensor) -> torch.Tensor:
        return self.margin - self(triples[:, 0], triples[:, 1], triples[:, 2])

    def filtered_loss(self, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        return F.relu(self.margin + self.score(positive) - self.score(negative)).mean()


class RotateV2(Rotate):
    """Original RotatE complex L1 score and scaled entity initialization."""

    def reset_parameters(self) -> None:
        super().reset_parameters()
        bound = (self.margin + 2.0) / self.hidden_channels
        torch.nn.init.uniform_(self.node_emb.weight, -bound, bound)
        torch.nn.init.uniform_(self.node_emb_im.weight, -bound, bound)

    def forward(self, head_index: torch.Tensor, rel_type: torch.Tensor, tail_index: torch.Tensor) -> torch.Tensor:
        head_re, head_im = self.node_emb(head_index), self.node_emb_im(head_index)
        tail_re, tail_im = self.node_emb(tail_index), self.node_emb_im(tail_index)
        phase = self.rel_emb(rel_type)
        cosine, sine = torch.cos(phase), torch.sin(phase)
        delta_re = cosine * head_re - sine * head_im - tail_re
        delta_im = cosine * head_im + sine * head_re - tail_im
        distance = torch.linalg.vector_norm(torch.stack((delta_re, delta_im), dim=-1), dim=-1).sum(dim=-1)
        return self.margin - distance

    def self_adversarial_loss(
        self, positive: torch.Tensor, negatives: torch.Tensor, temperature: float
    ) -> torch.Tensor:
        """Detached hard-negative weights with positive/negative log-sigmoid terms."""
        if negatives.ndim != 3 or negatives.shape[0] != positive.shape[0] or negatives.shape[2] != 3:
            raise ValueError("Expected negatives with shape [batch, count, 3]")
        positive_score = self(positive[:, 0], positive[:, 1], positive[:, 2])
        flattened = negatives.reshape(-1, 3)
        negative_score = self(flattened[:, 0], flattened[:, 1], flattened[:, 2]).reshape(negatives.shape[:2])
        weights = F.softmax(negative_score * temperature, dim=1).detach()
        positive_loss = -F.logsigmoid(positive_score).mean()
        negative_loss = -(weights * F.logsigmoid(-negative_score)).sum(dim=1).mean()
        return (positive_loss + negative_loss) / 2


def evaluate_rotate(model: Rotate, triples: np.ndarray, known: set[tuple[int, int, int]], device: torch.device) -> dict:
    """Filtered head/tail ranking, matching TransE's tie and known-positive rules."""
    head_known: dict[tuple[int, int], set[int]] = {}
    tail_known: dict[tuple[int, int], set[int]] = {}
    for h, r, t in known:
        head_known.setdefault((int(r), int(t)), set()).add(int(h))
        tail_known.setdefault((int(h), int(r)), set()).add(int(t))
    ranks = []
    model.eval()
    with torch.no_grad():
        real = model.node_emb.weight
        imag = model.node_emb_im.weight
        phases = model.rel_emb.weight
        for h, r, t in triples:
            h, r, t = int(h), int(r), int(t)
            cos, sin = torch.cos(phases[r]), torch.sin(phases[r])
            for side in (0, 2):
                if side == 0:
                    delta_real = cos * real - sin * imag - real[t]
                    delta_imag = cos * imag + sin * real - imag[t]
                    gold_id = h
                    filtered = head_known.get((r, t), ())
                else:
                    target_real = cos * real[h] - sin * imag[h]
                    target_imag = cos * imag[h] + sin * real[h]
                    delta_real = target_real - real
                    delta_imag = target_imag - imag
                    gold_id = t
                    filtered = tail_known.get((h, r), ())
                components = torch.linalg.vector_norm(torch.stack((delta_real, delta_imag), dim=-1), dim=-1)
                scores = (
                    components.sum(dim=1)
                    if isinstance(model, RotateV2)
                    else torch.linalg.vector_norm(components, dim=1)
                )
                gold = scores[gold_id]
                excluded = [index for index in filtered if index != gold_id]
                if excluded:
                    scores[torch.as_tensor(excluded, dtype=torch.long, device=device)] = float("inf")
                better = int(torch.count_nonzero(scores < gold - 1e-8).item())
                tied = int(torch.count_nonzero(torch.abs(scores - gold) <= 1e-8).item())
                ranks.append(1 + better + (tied - 1) / 2)
    return {
        "MRR": float(np.mean([1 / x for x in ranks])),
        "Hits@1": float(np.mean([x <= 1 for x in ranks])),
        "Hits@3": float(np.mean([x <= 3 for x in ranks])),
        "Hits@10": float(np.mean([x <= 10 for x in ranks])),
        "count": len(ranks),
    }
