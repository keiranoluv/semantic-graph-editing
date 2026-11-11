"""Test compact indices and collision-safe experiment directories."""
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from scripts.pilot_mlp import collate_training_graphs, create_run_dir


def graph(n, labels):
    return Data(
        x=torch.randn(n, 5),
        object_labels=torch.ones(n, dtype=torch.long),
        edge_label_index=torch.tensor([[0], [n - 1]]),
        edge_label=torch.tensor(labels),
        num_nodes=n,
    )


def test_skipped_graph_compact_indices():
    graphs = [graph(2, [1]), graph(9, [0]), graph(3, [2])]
    batch, skipped = collate_training_graphs(graphs, 0, torch.Generator().manual_seed(42))
    assert skipped == 1
    assert batch["pairs"].tolist() == [[0, 2], [1, 4]]
    assert batch["x"].shape[0] == 5


def test_separate_runs_never_overwrite(tmp_path: Path):
    first = create_run_dir(tmp_path, "baseline")
    (first / "best.pt").write_bytes(b"important")
    second = create_run_dir(tmp_path, "baseline")
    assert first != second
    assert (first / "best.pt").read_bytes() == b"important"


def test_invalid_run_name(tmp_path: Path):
    with pytest.raises(ValueError):
        create_run_dir(tmp_path, "../escape")
