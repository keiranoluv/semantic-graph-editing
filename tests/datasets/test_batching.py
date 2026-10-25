"""Data workers preserve image order and the model's random stream."""

import os
import pickle

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from src.datasets.batching import graph_batches, make_graph_loader
from src.datasets.hdf5_session import ProcessLocalH5


class IndexedDataset(Dataset):
    def __len__(self):
        return 7

    def __getitem__(self, index):
        return index, os.getpid()


def test_ordered_workers_keep_order_and_model_rng():
    dataset = IndexedDataset()
    before = torch.get_rng_state().clone()
    loader = make_graph_loader(dataset, 7, 3, num_workers=2, seed=1729)
    first = list(graph_batches(dataset, 7, 3, loader))
    second = list(graph_batches(dataset, 7, 3, loader))
    assert [begin for begin, _ in first] == [0, 3, 6]
    assert [index for _, batch in first for index, _ in batch] == list(range(7))
    assert [index for _, batch in second for index, _ in batch] == list(range(7))
    assert {pid for _, batch in first for _, pid in batch} == {pid for _, batch in second for _, pid in batch}
    assert torch.equal(before, torch.get_rng_state())


def test_process_local_hdf5_handle_reused_and_safe_to_pickle(tmp_path):
    path = tmp_path / "sample.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("value", data=[3])
        h5.create_dataset("compressed", data=np.arange(10), compression="gzip")
    reader = ProcessLocalH5(path)
    first = reader.get()
    assert reader.get() is first
    assert int(first["value"][0]) == 3
    with reader.open() as arrays:
        assert isinstance(arrays["value"], np.memmap)
        np.testing.assert_array_equal(arrays["value"], first["value"][:])
        assert isinstance(arrays["compressed"], h5py.Dataset)
        np.testing.assert_array_equal(arrays["compressed"][:], first["compressed"][:])
    restored = pickle.loads(pickle.dumps(reader))
    assert restored.get() is not first
    assert int(restored.get()["value"][0]) == 3
    reader.close()
    restored.close()
