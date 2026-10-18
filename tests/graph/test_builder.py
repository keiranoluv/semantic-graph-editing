"""Builder placeholder test; functional graph assertions are pending."""

import pytest
import torch

pytest.importorskip("torch_geometric")

from src.graph.builder import SceneGraphBuilder  # noqa: E402


def test_builder_placeholder(sample_scene):
    features = torch.zeros((len(sample_scene.objects), 4), dtype=torch.float32)
    with pytest.raises(NotImplementedError):
        SceneGraphBuilder().build(sample_scene, features)


@pytest.mark.skip(reason="Phase 4: assert 3-node directed edge_index shape [2, 6]")
def test_fully_connected_topology():
    pass


@pytest.mark.skip(reason="Phase 4: verify candidate predicate labels and background=0")
def test_candidate_pair_targets():
    pass
