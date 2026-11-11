"""Regression checks for GQA-specific multilabel and empty-graph semantics."""

import copy

import pytest
import torch

from src.datasets.gqa_pilot import select_cohort, validate_cohort
from src.datasets.schemas import SceneGraphRecord, SceneObject, SceneRelation
from src.evaluation.gqa_pilot import evaluate_gqa_image, summarize_gqa
from src.features.geometry import GeometryFeatureExtractor
from src.graph.builder import SceneGraphBuilder
from src.models.gat import GATPredicateClassifier
from src.models.mlp_semantic import SemanticMLPPredicateClassifier
from src.training.gqa_pilot import collate, cross_entropy_sum, forward, sample_pairs, soft_targets


def graph(predicates=(1, 1, 310), nodes=2):
    record = SceneGraphRecord(
        "example",
        100,
        100,
        tuple(SceneObject(i + 10, 1699, (i * 30.0, 0.0, i * 30.0 + 20.0, 20.0)) for i in range(nodes)),
        tuple(SceneRelation(10, 11, p) for p in predicates),
    )
    return SceneGraphBuilder().build(record, GeometryFeatureExtractor().extract(record))


def test_cohort_is_deterministic_disjoint_and_rejects_mutation():
    manifest = {"dataset": "fixture", "splits": {"train": list("abcdef"), "val": list("ghij"), "test": ["z"]}}
    cohort = select_cohort(manifest, "digest", 3, 2)
    reversed_manifest = copy.deepcopy(manifest)
    reversed_manifest["splits"]["train"].reverse()
    assert select_cohort(reversed_manifest, "digest", 3, 2) == cohort
    validate_cohort(cohort, manifest, "digest")
    assert "z" not in sum(cohort["splits"].values(), [])
    with pytest.raises(ValueError):
        validate_cohort(cohort, manifest, "other-digest")
    cohort["splits"]["train"][0] = "z"
    with pytest.raises(ValueError):
        validate_cohort(cohort, manifest, "digest")


def test_duplicate_rows_do_not_bias_uniform_multilabel_training():
    g = graph()
    target = soft_targets(g, 311)
    assert target.shape == (2, 311)
    assert target[0, 1] == target[0, 310] == 0.5
    assert target[0, 0] == 0 and target[1, 0] == 1
    assert torch.equal(target, soft_targets(graph((1, 310)), 311))
    logits = torch.zeros((2, 311), requires_grad=True)
    cross_entropy_sum(logits, target).backward()
    assert logits.grad[0, 1] < 0 and logits.grad[0, 310] < 0
    assert g.gt_predicate.numel() == 3
    with pytest.raises(ValueError):
        soft_targets(g, 310)


def test_sampling_retains_multilabel_and_handles_no_gt_and_empty():
    targets = soft_targets(graph(), 311)
    assert set(sample_pairs(targets, torch.Generator().manual_seed(42)).tolist()) == {0, 1}
    bg = soft_targets(graph((), 3), 311)
    assert len(sample_pairs(bg, torch.Generator().manual_seed(42))) == 6
    empty = soft_targets(graph((), 0), 311)
    assert empty.shape == (0, 311)
    assert len(sample_pairs(empty, torch.Generator().manual_seed(42))) == 0


@pytest.mark.parametrize("constrained,expected", [(True, 2 / 3), (False, 1.0)])
def test_eval_keeps_duplicates_multilabel_and_all_310_predicates(constrained, expected):
    g = graph()
    logits = torch.full((2, 311), -20.0)
    logits[:, 0] = 20.0
    logits[0, 1], logits[0, 310] = 30.0, 29.0
    result = evaluate_gqa_image(logits, g, 310, graph_constraint=constrained)
    summary = summarize_gqa([result], 310, graph_constraint=constrained)
    prefix = "" if constrained else "ng-"
    assert result["support"] == 3
    assert summary[f"{prefix}R@50"] == pytest.approx(expected)
    assert summary[f"{prefix}mR@50"] == pytest.approx((1 if constrained else 2) / 310)
    assert summary["num_predicates_for_mean"] == 310
    assert summary["num_predicates_with_gt"] == 2
    assert summary["per_predicate"]["310"]["support"] == 1
    with pytest.raises(ValueError):
        evaluate_gqa_image(logits, g, 50)


@pytest.mark.parametrize("nodes", [0, 1, 2])
@pytest.mark.parametrize("constrained", [True, False])
def test_no_gt_images_remain_reported_without_recall(nodes, constrained):
    g = graph((), nodes)
    result = evaluate_gqa_image(torch.zeros((nodes * (nodes - 1), 311)), g, 310, graph_constraint=constrained)
    summary = summarize_gqa([result], 310, graph_constraint=constrained)
    assert summary["num_images"] == 1 and summary["num_images_with_gt"] == 0
    assert summary["R@50" if constrained else "ng-R@50"] is None


@pytest.mark.parametrize("kind", ["mlp", "gat"])
def test_model_batch_matches_single_and_gt_cannot_enter_inputs(kind):
    torch.manual_seed(42)
    torch.set_num_threads(1)
    cls = GATPredicateClassifier if kind == "gat" else SemanticMLPPredicateClassifier
    model = cls(geometry_dim=5, num_object_classes=1700, hidden_dim=16, num_classes=311).eval()
    graphs = [graph(), graph((), 0), graph((), 1), graph((2,), 3)]
    batch, slices = collate(graphs, 311)
    predicted = forward(model, kind, batch, "cpu")
    for g, (start, end) in zip(graphs, slices, strict=True):
        single, _ = collate([g], 311)
        assert torch.allclose(predicted[start:end], forward(model, kind, single, "cpu"), atol=1e-6)
    changed = [graph((42,)), *graphs[1:]]
    other, _ = collate(changed, 311)
    assert torch.equal(forward(model, kind, other, "cpu"), predicted)
    generator = torch.Generator().manual_seed(42)
    sampled, _ = collate(graphs, 311, generator)
    assert torch.equal(sampled["edge_index"], batch["edge_index"])
    loss = cross_entropy_sum(forward(model, kind, sampled, "cpu"), sampled["targets"])
    loss.backward()
    assert torch.isfinite(loss)
