"""Derived supervision caches must preserve every batch and resume state exactly."""

import pytest
import torch
from torch_geometric.data import Data

from src.training import gqa_full
from src.training.gqa_pilot import collate
from tests.training.test_gqa_full import assert_nested_equal, fixtures
from tests.training.test_gqa_ram_loader import tiny_cache
from tools import run_gqa_full_ram as ram
from tools import train_gqa_pilot as evaluator
from tools.run_gqa_full_prepared import load_prepared_shard, prepare_supervision, prepared_collate


def test_prepared_batches_preserve_order_targets_and_sampler_state():
    torch.set_num_threads(1)
    pairs = torch.tensor([(i, j) for i in range(8) for j in range(8) if i != j]).T
    # Deliberately reverse candidate order, keep duplicates and three predicates.
    graph = Data(
        x=torch.zeros(8, 517),
        object_labels=torch.zeros(8, dtype=torch.long),
        edge_index=pairs,
        edge_label_index=pairs.flip(1),
        gt_relation_index=torch.tensor([[1, 1, 1, 1, 2], [2, 2, 2, 2, 3]]),
        gt_predicate=torch.tensor([1, 2, 3, 3, 4]),
        num_nodes=8,
    )
    no_gt = graph.clone()
    no_gt.gt_relation_index = torch.empty(2, 0, dtype=torch.long)
    no_gt.gt_predicate = torch.empty(0, dtype=torch.long)
    all_positive = graph.clone()
    all_positive.gt_relation_index = pairs.clone()
    all_positive.gt_predicate = torch.ones(pairs.shape[1], dtype=torch.long)
    empty = Data(
        x=torch.empty(0, 517),
        object_labels=torch.empty(0, dtype=torch.long),
        edge_index=torch.empty(2, 0, dtype=torch.long),
        edge_label_index=torch.empty(2, 0, dtype=torch.long),
        gt_relation_index=torch.empty(2, 0, dtype=torch.long),
        gt_predicate=torch.empty(0, dtype=torch.long),
        num_nodes=0,
    )
    graphs = [prepare_supervision(g, 7) for g in [graph, no_gt, empty, all_positive, graph]]
    for seed in range(32):
        left, right = [torch.Generator().manual_seed(seed) for _ in range(2)]
        assert_nested_equal(collate(graphs, 7, left), prepared_collate(graphs, 7, right))
        assert_nested_equal(left.get_state(), right.get_state())
    assert_nested_equal(collate(graphs, 7), prepared_collate(graphs, 7))
    with pytest.raises(ValueError, match="vocabulary"):
        prepared_collate(graphs, 8)
    with pytest.raises(ValueError, match="empty batch"):
        prepared_collate([], 7)


def test_spawned_workers_prepare_identical_supervision(tmp_path, monkeypatch):
    metadata = tiny_cache(tmp_path)
    monkeypatch.setattr(ram, "load_shard", load_prepared_shard)
    dataset = ram.RAMGQADataset(tmp_path, metadata, "train", workers=2)
    try:
        a, b = [torch.Generator().manual_seed(42) for _ in range(2)]
        assert_nested_equal(collate(dataset[:], 311, a), prepared_collate(dataset[:], 311, b))
        assert_nested_equal(collate(dataset[:], 311), prepared_collate(dataset[:], 311))
        assert_nested_equal(a.get_state(), b.get_state())
    finally:
        dataset.close()


@pytest.mark.parametrize("kind", gqa_full.MODELS)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_resume_with_prepared_collation_is_exact(tmp_path, monkeypatch, kind, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(1)
    train, val, vectors, experiment = fixtures(device)
    continuous, resumed = tmp_path / "continuous", tmp_path / "resumed"
    expected = gqa_full.train_run(kind, 42, train, val, vectors, experiment, continuous)
    assert gqa_full.train_run(kind, 42, train, val, vectors, experiment, resumed, stop_after_batches=1) is None
    for graph in train + val:
        prepare_supervision(graph, experiment["dimensions"]["num_classes"])
    monkeypatch.setattr(gqa_full, "collate", prepared_collate)
    monkeypatch.setattr(evaluator, "collate", prepared_collate)
    actual = gqa_full.train_run(kind, 42, train, val, vectors, experiment, resumed)
    for key in ("history", "validation", "initial_state_sha256", "selected_epoch"):
        assert_nested_equal(expected[key], actual[key])
    a, b = [torch.load(p / "last.pt", map_location="cpu", weights_only=True) for p in (continuous, resumed)]
    for key in ("model", "optimizer", "torch_rng", "cuda_rng", "sampler_rng", "best_model", "total_batches"):
        assert_nested_equal(a[key], b[key])
