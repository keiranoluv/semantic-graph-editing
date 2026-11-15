import pytest
import torch

from experiments.aligng_adapted_v1.model import AlignGConfig, AlignGPredicateClassifier
from src.datasets.schemas import SceneGraphRecord, SceneObject, SceneRelation
from src.graph.builder import SceneGraphBuilder


@pytest.fixture(autouse=True)
def cpu_determinism():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        yield
    torch.set_num_threads(previous)


def graph(nodes=4, predicates=(1, 1, 3), *, seed=42):
    record = SceneGraphRecord(
        str(seed),
        max(100, nodes + 10),
        max(100, nodes + 10),
        tuple(SceneObject(10 + i, i % 7 + 1, (i, i, i + 10, i + 10)) for i in range(nodes)),
        tuple(SceneRelation(10, 11, p) for p in predicates),
    )
    x = torch.randn(nodes, 517, generator=torch.Generator().manual_seed(seed))
    return SceneGraphBuilder().build(record, x)


def model(classes=51, *, dim=16, dropout=0.0, **flags):
    return AlignGPredicateClassifier(
        torch.randn(8, 12),
        torch.randn(classes, 12),
        semantic_dim=dim,
        dropout=dropout,
        config=AlignGConfig(**flags),
    )
