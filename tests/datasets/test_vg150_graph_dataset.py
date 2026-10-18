"""Integration tests for the VG150-to-PyG pipeline."""

from dataclasses import replace

import torch

from src.datasets.vg150 import VG150GraphDataset
from src.features.geometry import GeometryFeatureExtractor
from src.graph.builder import SceneGraphBuilder


class FakeAnnotationLoader:
    """In-memory annotation loader for integration testing."""

    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def load(self, index):
        return self.records[index]


def make_dataset(records):
    """Build a graph dataset with real feature and graph components."""
    return VG150GraphDataset(
        loader=FakeAnnotationLoader(records),
        feature_extractor=GeometryFeatureExtractor(),
        graph_builder=SceneGraphBuilder(),
    )


def test_end_to_end_graph(sample_scene):
    """One synthetic annotation becomes a valid PyG graph."""
    dataset = make_dataset([sample_scene])

    assert len(dataset) == 1

    graph = dataset[0]

    assert graph.num_nodes == 3
    assert graph.x.shape == (3, 5)
    assert graph.x.dtype == torch.float32
    assert graph.edge_index.shape == (2, 6)
    assert graph.edge_label_index.shape == (2, 6)
    assert graph.edge_label.shape == (6,)
    assert graph.gt_predicate.tolist() == [1]
    assert graph.object_ids.tolist() == [10, 30, 20]
    assert graph.image_id == "synthetic_001"


def test_dataset_indexing(sample_scene):
    """Different indices produce the corresponding image graphs."""
    second = replace(
        sample_scene,
        image_id="synthetic_002",
        relations=(),
    )
    dataset = make_dataset([sample_scene, second])

    assert len(dataset) == 2
    assert dataset[0].image_id == "synthetic_001"
    assert dataset[1].image_id == "synthetic_002"


def test_empty_dataset():
    """A dataset may contain no selected images."""
    dataset = make_dataset([])

    assert len(dataset) == 0


def test_message_passing_independent_of_labels(sample_scene):
    """Removing annotations must not change message-passing edges."""
    without_relations = replace(sample_scene, relations=())
    dataset = make_dataset([sample_scene, without_relations])

    graph_a = dataset[0]
    graph_b = dataset[1]

    assert torch.equal(graph_a.edge_index, graph_b.edge_index)
    assert graph_a.gt_predicate.numel() == 1
    assert graph_b.gt_predicate.numel() == 0
