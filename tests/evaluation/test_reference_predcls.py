"""Regression cases for reference semantics, independent of historical metrics."""

from collections import Counter

import pytest
import torch
from torch_geometric.data import Data

from src.evaluation.batched import evaluate_graph_batch
from src.evaluation.gat import evaluate_gat_batch
from src.evaluation.reference_predcls import box_iou, evaluate_reference_image, summarize_reference
from src.evaluation.scene_graph import evaluate_image, summarize_evaluation
from src.models.gat import GATPredicateClassifier
from src.models.mlp_semantic import SemanticMLPPredicateClassifier


def test_image_macro_and_absent_classes():
    rows = [
        {"support": n, "hits": {50: h}, "class_support": Counter({1: n}), "class_hits": {50: Counter({1: h})}}
        for n, h in [(9, 9), (1, 0)]
    ]
    assert summarize_evaluation(rows, (50,))["mR@50"] == 0.9
    summary = summarize_reference(rows, 2, (50,))
    assert summary["mR@50"] == 0.25
    assert summary["R@50"] == 0.5
    assert summary["per_predicate"]["1"]["R@50"] == 0.5
    assert summary["per_predicate"]["2"]["R@50"] == 0


def test_overlapping_instances_duplicates_and_multiple_labels():
    boxes = torch.tensor([[0.0, 0.0, 9.0, 9.0], [0.0, 0.0, 9.0, 9.0], [20.0, 20.0, 29.0, 29.0]])
    classes = torch.tensor([1, 1, 2])
    pairs = torch.tensor([[1], [2]])
    gt = torch.tensor([[0, 0, 0], [2, 2, 2]])
    predicates = torch.tensor([1, 1, 2])
    logits = torch.tensor([[0.0, 9.0, 1.0]])
    result = evaluate_reference_image(logits, pairs, gt, predicates, boxes, classes)
    assert result["support"] == 3  # Retain duplicate annotations.
    assert result["hits"][50] == 2  # One prediction matches two annotation rows.
    assert evaluate_image(logits, pairs, gt, predicates)["hits"][50] == 0
    classes[1] = 3
    assert evaluate_reference_image(logits, pairs, gt, predicates, boxes, classes)["hits"][50] == 0


def test_inclusive_iou_boundary():
    # Inclusive areas 2 and 1 -> IoU 0.5; continuous coordinates would degenerate.
    boxes = torch.tensor([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    assert box_iou(boxes)[0, 1] == 0.5


def test_ranking_uses_full_softmax_and_one_foreground():
    logits = torch.tensor([[100.0, 9.0, 1.0], [0.0, 2.0, 1.0]])
    pairs = torch.tensor([[0, 1], [2, 2]])
    boxes = torch.tensor([[0.0, 0.0, 5.0, 5.0], [10.0, 10.0, 15.0, 15.0], [20.0, 20.0, 25.0, 25.0]])
    result = evaluate_reference_image(
        logits, pairs, pairs[:, :1], torch.tensor([1]), boxes, torch.tensor([1, 2, 3]), ks=(1, 2)
    )
    assert result["hits"] == {1: 0, 2: 1}


@pytest.mark.parametrize("graph_constraint", [True, False])
def test_empty_candidates_and_no_gt(graph_constraint):
    result = evaluate_reference_image(
        torch.empty(0, 3),
        torch.empty(2, 0, dtype=torch.long),
        torch.empty(2, 0, dtype=torch.long),
        torch.empty(0, dtype=torch.long),
        torch.empty(0, 4),
        torch.empty(0, dtype=torch.long),
        graph_constraint=graph_constraint,
    )
    summary = summarize_reference([result], 2)
    assert summary["R@50"] is None and summary["mR@50"] is None
    assert summarize_reference([], 2)["num_images"] == 0


@pytest.mark.parametrize("name", ["gat", "mlp_semantic"])
@pytest.mark.parametrize("graph_constraint", [True, False])
def test_reference_batching_preserves_image_boundaries(name, graph_constraint):
    torch.manual_seed(42)
    graphs = []
    for n in [3, 5]:
        pairs = torch.tensor([(s, o) for s in range(n) for o in range(n) if s != o]).T
        graphs.append(
            Data(
                x=torch.randn(n, 5),
                object_labels=torch.arange(1, n + 1),
                boxes=torch.tensor([[i * 10.0, 0.0, i * 10.0 + 4.0, 4.0] for i in range(n)]),
                edge_index=pairs,
                edge_label_index=pairs,
                edge_label=torch.zeros(pairs.shape[1], dtype=torch.long),
                gt_relation_index=pairs[:, :1],
                gt_predicate=torch.tensor([1]),
                num_nodes=n,
            )
        )
    model = (
        GATPredicateClassifier(hidden_dim=16, num_heads=2, dropout=0.0)
        if name == "gat"
        else SemanticMLPPredicateClassifier(hidden_dim=16)
    ).eval()
    from scripts.extract_level1_examples import predict_graph

    with torch.inference_mode():
        expected = [
            evaluate_reference_image(
                predict_graph(model, name, g, torch.device("cpu")),
                g.edge_label_index,
                g.gt_relation_index,
                g.gt_predicate,
                g.boxes,
                g.object_labels,
                graph_constraint=graph_constraint,
            )
            for g in graphs
        ]
        protocol = "reference-v1" if graph_constraint else "reference-ng-v1"
        if name == "gat":
            actual, _, _ = evaluate_gat_batch(model, graphs, torch.device("cpu"), protocol=protocol)
        else:
            actual, _, _ = evaluate_graph_batch(model, graphs, name, torch.device("cpu"), protocol=protocol)
    assert actual == expected


def test_no_constraint_recovers_multiple_predicates_and_keeps_duplicate_gt_rows():
    pairs = torch.tensor([[0], [1]])
    boxes = torch.tensor([[0.0, 0.0, 4.0, 4.0], [10.0, 0.0, 14.0, 4.0]])
    args = (
        torch.tensor([[0.0, 3.0, 2.0]]),
        pairs,
        pairs.repeat(1, 3),
        torch.tensor([1, 1, 2]),
        boxes,
        torch.tensor([1, 2]),
    )
    gc = evaluate_reference_image(*args, ks=(1, 2))
    ng = evaluate_reference_image(*args, ks=(1, 2), graph_constraint=False)
    assert gc["hits"] == {1: 2, 2: 2}
    assert ng["hits"] == {1: 2, 2: 3}
    summary = summarize_reference([ng], 2, (1, 2), graph_constraint=False)
    assert summary["ng-R@2"] == summary["ng-mR@2"] == 1
    assert summary["ng-mR@1"] == 0.5
    assert "R@2" not in summary


def test_no_constraint_global_budget_can_displace_another_correct_pair():
    # Two high-scoring predicates on pair 0 consume both slots; only its first
    # predicate is GT. Pair 1's correct predicate fits GC top-2, but not NG top-2.
    logits = torch.tensor([[0.0, 5.0, 4.9], [2.0, 2.0, 1.0]])
    pairs = torch.tensor([[0, 1], [2, 2]])
    boxes = torch.tensor([[0.0, 0.0, 4.0, 4.0], [10.0, 0.0, 14.0, 4.0], [20.0, 0.0, 24.0, 4.0]])
    args = (logits, pairs, pairs, torch.tensor([1, 1]), boxes, torch.tensor([1, 2, 3]))
    assert evaluate_reference_image(*args, ks=(2,))["hits"][2] == 2
    assert evaluate_reference_image(*args, ks=(2,), graph_constraint=False)["hits"][2] == 1
