"""Ensure subgroup diagnostics keep original ranking separate from interventions."""

import torch
from torch_geometric.data import Data

from src.evaluation.gqa_pilot import summarize_gqa
from src.training.gqa_pilot import soft_targets
from tools.diagnose_gqa_quality import diagnose_image, grouped_metrics


def test_excluding_gt_does_not_silently_rerank_but_counterfactual_does():
    pairs = torch.tensor([(i, j) for i in range(9) for j in range(9) if i != j]).T
    boxes = torch.tensor([[i * 20, 0, i * 20 + 10, 10] for i in range(9)], dtype=torch.float32)
    g = Data(
        edge_label_index=pairs,
        gt_relation_index=pairs[:, [0, 0, 71]],
        gt_predicate=torch.tensor([1, 1, 3]),
        boxes=boxes,
        object_labels=torch.arange(1, 10),
    )
    logits = torch.full((72, 5), -6.0)
    logits[:, 0] = 0
    logits[:, 1] = 10
    logits[71, 1] = -10
    logits[71, 3] = 5
    before = logits.clone()
    row, masked, predictions, stats = diagnose_image(logits, g, soft_targets(g, 5), {1, 2})
    standard = summarize_gqa([row], 4)
    intervened = summarize_gqa([masked], 4)
    original_group = grouped_metrics([row], standard, [3, 4])
    masked_group = grouped_metrics([masked], intervened, [3, 4])
    assert original_group["R@50"] == 0
    assert masked_group["R@50"] == 1
    assert masked_group["mR@50"] == 0.5  # absent fourth class remains zero
    assert masked_group["mR_supported@50"] == 1
    assert row["support"] == 3 and masked["support"] == 1
    assert predictions[1] == 50
    assert stats["gt_unique_pairs"] == 2
    assert stats["gt_pairs_correct_foreground"] == 2
    torch.testing.assert_close(logits, before, rtol=0, atol=0)


def test_empty_group_has_no_recall_denominator():
    summary = {"per_predicate": {"1": {"support": 0, "R@50": 0.0, "R@100": 0.0}}}
    value = grouped_metrics([], summary, [1])
    assert value["R@50"] is None
    assert value["mR_supported@50"] is None
    assert value["mR@50"] == 0
