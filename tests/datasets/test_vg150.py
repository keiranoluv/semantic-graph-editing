"""VG150 public API contract tests."""

from src.datasets.vg150 import VG150AnnotationLoader, VG150GraphDataset


def test_loader_interface():
    """Verify the annotation loader exposes its public methods."""
    assert callable(VG150AnnotationLoader.load)
    assert callable(VG150AnnotationLoader.__len__)


def test_graph_dataset_interface():
    """Verify the graph dataset remains a separate adapter."""
    assert callable(VG150GraphDataset.get)
    assert callable(VG150GraphDataset.len)
