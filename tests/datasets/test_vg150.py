"""Dataset skeletal-contract tests, not integration tests."""

from pathlib import Path

import pytest

pytest.importorskip("torch_geometric")

from src.datasets.vg150 import VG150AnnotationLoader  # noqa: E402


def test_loader_is_explicitly_unimplemented():
    loader = VG150AnnotationLoader(Path("unused"), "train", Path("split.json"))
    with pytest.raises(NotImplementedError, match="source format"):
        loader.load(0)
