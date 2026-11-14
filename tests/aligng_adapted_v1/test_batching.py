import copy

import pytest
import torch

from experiments.aligng_adapted_v1.batching import collate, context_pairs, inference_batch, model_inputs
from src.training.gqa_pilot import collate as baseline_collate

from .conftest import graph


@pytest.mark.parametrize("classes", [51, 311])
@pytest.mark.parametrize("sampled", [True, False])
def test_sampler_targets_offsets_match_frozen_baseline(classes, sampled):
    graphs = [graph(5, (1, 1, classes - 1)), graph(0, ()), graph(1, ()), graph(4, (), seed=17)]
    generators = [torch.Generator().manual_seed(123) if sampled else None for _ in range(2)]
    expected, slices = baseline_collate(graphs, classes, generators[0])
    actual = collate(graphs, classes, generators[1])
    for old, new in (
        ("x", "x"),
        ("object_labels", "object_labels"),
        ("pairs", "query_pairs"),
        ("targets", "targets"),
        ("edge_index", "context_pairs"),
    ):
        assert torch.equal(expected[old], actual[new])
    assert list(zip(actual["query_ptr"][:-1].tolist(), actual["query_ptr"][1:].tolist(), strict=True)) == slices
    if sampled:
        assert torch.equal(generators[0].get_state(), generators[1].get_state())
    positive = actual["targets"][0]
    assert positive[1] == positive[-1] == 0.5 and positive.sum() == 1


def test_context_independent_of_labels_sampler_and_duplicate_annotations():
    original = graph(7)
    changed = graph(7, (4, 9, 9))
    assert torch.equal(original.x, changed.x)
    batches = [
        collate([g], 51, torch.Generator().manual_seed(s), negative_ratio=r)
        for g, s, r in ((original, 42, 3), (changed, 123, 0), (changed, 3407, 8))
    ]
    for b in batches:
        assert torch.equal(b["context_pairs"], original.edge_index)
        assert b["context_ptr"].tolist() == [0, 42]
        assert "targets" not in model_inputs(b)
    assert not torch.equal(batches[0]["query_pairs"], batches[1]["query_pairs"])
    # No GT attributes are required for full-context inference.
    del changed.gt_predicate
    del changed.gt_relation_index
    assert torch.equal(context_pairs(changed), original.edge_index)
    assert torch.equal(inference_batch([changed])["context_pairs"], original.edge_index)


def test_duplicate_annotations_do_not_reweight_targets():
    a, b = collate([graph(predicates=(1, 1, 3))], 51), collate([graph(predicates=(1, 3))], 51)
    assert torch.equal(a["targets"], b["targets"])
    assert a["targets"][0, 1] == a["targets"][0, 3] == 0.5


@pytest.mark.parametrize(
    "mutation", ["sampled_context", "duplicate_context", "self_loop", "outside", "sampled_candidates"]
)
def test_incomplete_or_invalid_topology_rejected(mutation):
    g = copy.deepcopy(graph())
    if mutation == "sampled_context":
        g.edge_index = g.edge_index[:, :2]
    elif mutation == "duplicate_context":
        g.edge_index[:, 1] = g.edge_index[:, 0]
    elif mutation == "self_loop":
        g.edge_index[:, 0] = 0
    elif mutation == "outside":
        g.edge_index[0, 0] = 4
    else:
        g.edge_label_index = g.edge_label_index[:, :2]
    with pytest.raises(ValueError):
        collate([g], 51)


def test_no_graphs_rejected_but_zero_node_graph_valid():
    with pytest.raises(ValueError):
        collate([], 51)
    with pytest.raises(ValueError):
        inference_batch([])
    batch = collate([graph(0, ())], 311)
    assert batch["targets"].shape == (0, 311)
    assert batch["node_ptr"].tolist() == batch["context_ptr"].tolist() == batch["query_ptr"].tolist() == [0, 0]
