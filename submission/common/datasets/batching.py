"""Ordered graph batches with optional persistent CPU data workers."""

from collections.abc import Iterator

import torch
from torch.utils.data import DataLoader, Subset


def _list_collate(graphs):
    return graphs


def make_graph_loader(dataset, limit: int, batch_size: int, num_workers: int, seed: int) -> DataLoader:
    """Load immutable graphs in split order; keep model/sampler RNG separate."""
    if not 0 < limit <= len(dataset) or batch_size <= 0 or num_workers < 0:
        raise ValueError("Invalid graph loader size or worker count")
    options = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "collate_fn": _list_collate,
        "generator": torch.Generator().manual_seed(seed),
    }
    if num_workers:
        options.update(
            persistent_workers=True,
            prefetch_factor=2,
            pin_memory=True,
            multiprocessing_context="forkserver",
        )
    return DataLoader(Subset(dataset, range(limit)), **options)


def graph_batches(dataset, limit: int, batch_size: int, loader=None) -> Iterator[tuple[int, list]]:
    """Yield the same image batch boundaries with or without data workers."""
    begins = range(0, limit, batch_size)
    if loader is None:
        for begin in begins:
            yield begin, [dataset[i] for i in range(begin, min(begin + batch_size, limit))]
    else:
        for begin, graphs in zip(begins, loader, strict=True):
            yield begin, graphs
