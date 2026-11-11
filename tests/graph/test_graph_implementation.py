"""Functional tests for the synthetic scene graph pipeline."""

from dataclasses import replace

import pytest
import torch

pytest.importorskip("torch_geometric")

from src.datasets.schemas import SceneRelation  # noqa: E402
from src.features.geometry import GeometryFeatureExtractor  # noqa: E402
from src.graph.builder import SceneGraphBuilder  # noqa: E402


def test_geometry_features(sample_scene):
    extractor = GeometryFeatureExtractor()
    features = extractor.extract(sample_scene)

    assert features.shape == (3, 5)
    assert features.dtype == torch.float32
    assert torch.isfinite(features).all()
    assert torch.all(features >= 0)
    assert torch.all(features <= 1)


def test_graph_shapes(sample_scene):
    features = GeometryFeatureExtractor().extract(sample_scene)
    graph = SceneGraphBuilder().build(sample_scene, features)

    assert graph.num_nodes == 3
    assert graph.x.shape == (3, 5)
    assert graph.edge_index.shape == (2, 6)
    assert graph.edge_label_index.shape == (2, 6)
    assert graph.edge_label.shape == (6,)
    assert graph.boxes.shape == (3, 4)
    assert graph.object_labels.shape == (3,)


def test_object_id_mapping(sample_scene):
    features = GeometryFeatureExtractor().extract(sample_scene)
    graph = SceneGraphBuilder().build(sample_scene, features)

    assert graph.object_ids.tolist() == [10, 30, 20]
    assert graph.gt_relation_index.tolist() == [[0], [1]]
    assert graph.gt_predicate.tolist() == [1]


def test_ground_truth_independent_topology(sample_scene):
    extractor = GeometryFeatureExtractor()
    builder = SceneGraphBuilder()

    no_relations = replace(sample_scene, relations=())

    graph_a = builder.build(sample_scene, extractor.extract(sample_scene))
    graph_b = builder.build(
        no_relations,
        extractor.extract(no_relations),
    )

    assert torch.equal(graph_a.edge_index, graph_b.edge_index)
    assert graph_a.edge_label.tolist() == [1, 0, 0, 0, 0, 0]
    assert graph_b.edge_label.tolist() == [0] * 6


def test_multiple_predicates_are_preserved(sample_scene):
    multi = replace(
        sample_scene,
        relations=(
            SceneRelation(10, 30, 1),
            SceneRelation(10, 30, 2),
        ),
    )

    features = GeometryFeatureExtractor().extract(multi)
    graph = SceneGraphBuilder().build(multi, features)

    assert graph.gt_predicate.tolist() == [1, 2]
    assert graph.gt_relation_index.tolist() == [[0, 0], [1, 1]]
    assert graph.edge_label[0].item() == -1


def test_invalid_feature_shape(sample_scene):
    invalid = torch.zeros((2, 5), dtype=torch.float32)

    with pytest.raises(ValueError, match="shape"):
        SceneGraphBuilder().build(sample_scene, invalid)
