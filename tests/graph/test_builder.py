"""Smoke tests for scene graph construction."""

import pytest
import torch

pytest.importorskip("torch_geometric")

from src.graph.builder import SceneGraphBuilder  # noqa: E402


def test_builder_returns_graph(sample_scene):
    features = torch.zeros(
        (len(sample_scene.objects), 5),
        dtype=torch.float32,
    )

    graph = SceneGraphBuilder().build(sample_scene, features)

    assert graph.num_nodes == 3
    assert graph.edge_index.shape == (2, 6)
    assert graph.gt_predicate.tolist() == [1]
