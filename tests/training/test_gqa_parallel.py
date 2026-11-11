"""Validation reuse and cross-process coordination must retain exact behavior."""

import multiprocessing as mp
import os
from multiprocessing.reduction import DupFd

import pytest
import torch

from src.training import gqa_full
from tests.training.test_gqa_full import assert_nested_equal, fixtures
from tools import train_gqa_pilot as evaluator
from tools.run_gqa_full import exclusive
from tools.run_gqa_full_parallel import ValidationCache, coordinator_lock
from tools.run_gqa_full_prepared import prepare_supervision, prepared_collate


def retain_lock(fd, ready, finish):
    handle = fd.detach()
    ready.set()
    finish.wait(20)
    os.close(handle)


def test_child_keeps_root_lock_if_coordinator_descriptor_closes(tmp_path):
    ctx = mp.get_context("spawn")
    ready, finish = ctx.Event(), ctx.Event()
    with coordinator_lock(tmp_path) as fd:
        child = ctx.Process(target=retain_lock, args=(DupFd(fd), ready, finish))
        child.start()
        assert ready.wait(15)
        with pytest.raises(RuntimeError, match="Another process"):
            with exclusive(tmp_path):
                pass
    try:
        with pytest.raises(RuntimeError, match="Another process"):
            with exclusive(tmp_path):
                pass
    finally:
        finish.set()
        child.join(15)
    assert child.exitcode == 0
    with exclusive(tmp_path), pytest.raises(RuntimeError, match="Another process"):
        with coordinator_lock(tmp_path):
            pass


def test_validation_cache_is_bounded_and_does_not_cache_training():
    train, val, _, _ = fixtures("cpu")
    for graph in train + val:
        prepare_supervision(graph, 4)
    cache = ValidationCache()
    expected = prepared_collate(val, 4)
    actual = cache(val, 4)
    assert_nested_equal(expected, actual)
    assert cache(val, 4) is actual
    assert cache.hits == 1
    assert cache.bytes == sum(t.numel() * t.element_size() for t in actual[0].values())
    a, b = [torch.Generator().manual_seed(42) for _ in range(2)]
    assert_nested_equal(cache(train, 4, a), prepared_collate(train, 4, b))
    assert_nested_equal(a.get_state(), b.get_state())
    assert len(cache.batches) == 1
    bounded = ValidationCache(max_bytes=0)
    assert_nested_equal(bounded(val, 4), expected)
    assert bounded.bytes == 0 and not bounded.batches


@pytest.mark.parametrize("kind", gqa_full.MODELS)
def test_cached_validation_does_not_change_training_or_mutate_batches(tmp_path, monkeypatch, kind):
    torch.set_num_threads(1)
    train, val, vectors, experiment = fixtures("cpu")
    original = gqa_full.train_run(kind, 42, train, val, vectors, experiment, tmp_path / "reference")
    for graph in train + val:
        prepare_supervision(graph, 4)
    cache = ValidationCache()
    before = {k: t.clone() for k, t in cache(val, 4)[0].items()}
    monkeypatch.setattr(gqa_full, "collate", prepared_collate)
    monkeypatch.setattr(evaluator, "collate", cache)
    actual = gqa_full.train_run(kind, 42, train, val, vectors, experiment, tmp_path / "cached")
    assert_nested_equal(original["history"], actual["history"])
    assert_nested_equal(original["validation"], actual["validation"])
    assert_nested_equal(before, cache(val, 4)[0])
    assert cache.hits >= 3
