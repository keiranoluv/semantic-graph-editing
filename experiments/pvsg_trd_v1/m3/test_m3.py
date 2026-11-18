"""CPU regression tests for data leakage, sparse math, gradients and recovery."""

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from ..decoding import WindowLogits, decode_windows
from ..graphs import build_graph
from ..intervals import window_bounds
from ..m2.geometry import geometry_from_mask, motion_from_geometry
from ..m2.preparation import rasterize_adapted_targets
from ..synthetic import synthetic_masks, synthetic_video_targets
from .common import config, seed_all, write_json
from .data import FitSampler, Windows, collate, make_window
from .decoding import Accumulator
from .engine import backward_microbatch, denominators, restore_checkpoint, save_checkpoint, should_stop
from .models import RelationModel
from .probe import choose_batch, overfit


class FixtureStore:
    def __init__(self):
        video, target, self.geometry, self.visual = fixture()
        self.videos = {video["video_id"]: video}
        self.targets = {video["video_id"]: target}
        self.ignored = {video["video_id"]: []}

    def ids(self, split):
        assert split in ("fit", "validation")
        return list(self.videos)

    def window(self, vid, start, size, *, supervision=True):
        return make_window(
            self.videos[vid],
            self.geometry,
            self.visual,
            start,
            min(start + size, self.videos[vid]["num_frames"]),
            target=self.targets[vid] if supervision else None,
        )


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(2)


def fixture():
    video, target = synthetic_video_targets()
    ids = np.array([t["track_id"] for t in video["tracks"]])
    masks = synthetic_masks()
    results = [geometry_from_mask(mask, ids) for mask in masks]
    geometry = dict(
        track_ids=ids,
        class_ids=np.array([t["class_id"] for t in video["tracks"]]),
        boxes=np.stack([r[0] for r in results], axis=1),
        geometry=np.stack([r[1] for r in results], axis=1),
        visible=np.stack([r[2] for r in results], axis=1),
        rgb_available=np.ones(10, dtype=bool),
    )
    geometry["motion"], geometry["delta_frames"] = motion_from_geometry(geometry["geometry"], geometry["visible"])
    positions = np.argwhere(geometry["visible"].T)
    features = np.random.default_rng(17).normal(size=(len(positions), 512)).astype(np.float32)
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    visual = dict(
        features=features,
        frame_ids=positions[:, 0],
        track_ids=ids[positions[:, 1]],
        visual_available=np.ones(len(positions), dtype=bool),
    )
    return video, target, geometry, visual


def small_config():
    cfg = config()
    cfg.update(hidden_dim=16, class_embedding_dim=8, heads=2, blocks=1, dropout=0.0, pair_chunk_size=13)
    return cfg


def sample(start=0, stop=10):
    video, target, geometry, visual = fixture()
    return make_window(video, geometry, visual, start, stop, target=target)


def test_vector_graph_and_sparse_targets_equal_locked_M1_M2():
    video, target, geometry, visual = fixture()
    ignored = [dict(subject_id=7, object_id=12, predicate_id=0)]
    for start, stop in [(0, 10), (3, 8), (5, 7)]:
        s = make_window(video, geometry, visual, start, stop, target=target, ignored=ignored)
        expected = build_graph(video, start=start, stop=stop)
        for field in ("nodes", "spatial_edges", "temporal_edges", "self_edges", "query_edges"):
            np.testing.assert_array_equal(getattr(s["graph"], field), getattr(expected, field))
        dense = rasterize_adapted_targets(video, target, ignored, start=start, stop=stop)
        pairs = {tuple(p): i for i, p in enumerate(dense["pairs"])}
        for i, (a, b) in enumerate(s["graph"].query_edges.T):
            t, subject, _ = s["graph"].nodes[a]
            obj = s["graph"].nodes[b, 1]
            np.testing.assert_array_equal(s["labels"][i], dense["labels"][pairs[subject, obj], t - start])
            np.testing.assert_array_equal(
                s["supervision_mask"][i], dense["supervision_mask"][pairs[subject, obj], t - start]
            )


@pytest.mark.parametrize("kind", ["mlp", "static_gat", "st_gat"])
def test_batch_isolation_and_track_ID_not_a_feature(kind):
    seed_all(17)
    model = RelationModel(kind, small_config()).eval()
    a, b = sample(0, 4), sample(4, 10)
    with torch.no_grad():
        separate = torch.cat([model(collate([a])), model(collate([b]))])
        together = model(collate([a, b]))
        renamed = collate([a, b])
        renamed["nodes"][:, 1] += 100000
        torch.testing.assert_close(separate, together, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(together, model(renamed), atol=0, rtol=0)


@pytest.mark.parametrize("kind", ["mlp", "static_gat", "st_gat"])
def test_chunk_proxy_backward_equals_full_head_and_uneven_accumulation(kind):
    seed_all(17)
    reference = RelationModel(kind, small_config())
    chunked, accumulated = copy.deepcopy(reference), copy.deepcopy(reference)
    parts = [sample(0, 4), sample(2, 7), sample(5, 10)]
    batch = collate(parts)
    logits = reference(batch)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch["labels"], reduction="none")
    loss = ((loss * batch["supervision_mask"]).sum(1) / denominators(batch)[batch["query_batch"]]).sum() / 3
    loss.backward()
    backward_microbatch(chunked, batch, chunk_size=7, group_windows=3)
    backward_microbatch(accumulated, collate(parts[:2]), chunk_size=9, group_windows=3)
    backward_microbatch(accumulated, collate(parts[2:]), chunk_size=9, group_windows=3)
    for a, b, c in zip(reference.parameters(), chunked.parameters(), accumulated.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad, atol=2e-7, rtol=2e-4)
        torch.testing.assert_close(a.grad, c.grad, atol=2e-7, rtol=2e-4)


@pytest.mark.parametrize("kind,has_temporal", [("mlp", False), ("static_gat", False), ("st_gat", True)])
def test_only_ST_GAT_passes_visual_information_across_frames(kind, has_temporal):
    seed_all(17)
    batch = collate([sample(0, 4)])
    batch["visual"].requires_grad_(True)
    model = RelationModel(kind, small_config()).eval()
    row = torch.nonzero(batch["query_frame_indices"] == 1)[0, 0]
    model(batch)[row].sum().backward()
    previous = batch["visual"].grad[batch["nodes"][:, 0] == 0].abs().sum().item()
    assert (previous > 1e-8) == has_temporal


def test_sparse_streaming_decoder_equals_M1_for_overlaps_and_visibility_gaps():
    video, _, geometry, visual = fixture()
    merger = Accumulator(video)
    windows = []
    ids = np.array([t["track_id"] for t in video["tracks"]])
    rng = np.random.default_rng(4)
    for start, stop in window_bounds(10, 6, 4):
        s = make_window(video, geometry, visual, start, stop)
        pairs = np.array([(a, b) for a in ids for b in ids if a != b])
        dense = rng.normal(size=(len(pairs), stop - start, 57)).astype(np.float32)
        index = {tuple(p): i for i, p in enumerate(pairs)}
        sparse = []
        for a, b in s["graph"].query_edges.T:
            t, subject, _ = s["graph"].nodes[a]
            sparse.append(dense[index[subject, s["graph"].nodes[b, 1]], t - start])
        merger.add(s["graph"], np.asarray(sparse), start)
        windows.append(WindowLogits(pairs, np.arange(start, stop), dense))
    actual, expected = merger.finish(), decode_windows(video, windows)
    assert actual == expected


def test_missing_window_predictions_are_rejected():
    video, _, _, _ = fixture()
    merger = Accumulator(video)
    with pytest.raises(ValueError, match="every covisible"):
        merger.finish()


def test_batch_selection_uses_headroom_and_throughput_plateau():
    rows = [
        dict(
            state="passed",
            batch_size=size,
            heavy=dict(peak_reserved_bytes=memory),
            mixed=dict(peak_reserved_bytes=memory, windows_per_second=speed),
        )
        for size, memory, speed in [(1, 10, 10), (2, 20, 20), (4, 40, 20.5), (8, 90, 100)]
    ]
    assert choose_batch(rows, 80) == 2
    with pytest.raises(ValueError):
        choose_batch(rows, 5)


def test_checkpoint_resume_keeps_optimizer_and_RNG_and_rejects_drift(tmp_path):
    cfg = small_config()
    seed_all(17)
    model = RelationModel("mlp", cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt)
    backward_microbatch(model, collate([sample()]), chunk_size=13, group_windows=1)
    opt.step()
    opt.zero_grad(set_to_none=True)
    path = tmp_path / "test.pt"
    try:
        save_checkpoint(
            path,
            dict(
                signature="fixed",
                model=model.state_dict(),
                optimizer=opt.state_dict(),
                scheduler=sched.state_dict(),
                torch_rng=torch.get_rng_state(),
                cuda_rng=[],
            ),
        )
        expected_rng = torch.rand(5)
        new = RelationModel("mlp", cfg)
        new_opt = torch.optim.AdamW(new.parameters(), lr=0.1)
        new_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(new_opt)
        restore_checkpoint(path, new, new_opt, new_sched, "fixed", "cpu")
        torch.testing.assert_close(torch.rand(5), expected_rng)
        assert new_opt.param_groups[0]["lr"] == 0.001
        torch.testing.assert_close(model(collate([sample()])), new(collate([sample()])))
        with pytest.raises(ValueError, match="differ"):
            restore_checkpoint(path, new, new_opt, new_sched, "wrong", "cpu")
    finally:
        path.unlink(missing_ok=True)
    assert not list(tmp_path.rglob("*.pt"))


def test_early_stopping_respects_minimum_epoch_budget():
    cfg = config()
    assert not should_stop(9, 15, cfg)
    assert not should_stop(12, 9, cfg)
    assert should_stop(12, 10, cfg)


def test_incomplete_M2_blocks_main_training_before_GPU(tmp_path):
    from .preflight import check_features

    cfg = config()
    cfg["feature_cache"] = str(tmp_path)
    with pytest.raises(ValueError, match="M2 extraction is incomplete"):
        check_features(cfg)
    assert not torch.cuda.is_initialized()


def test_sampler_and_spawn_workers_are_epoch_reproducible():
    store = FixtureStore()
    sampler = FitSampler(store, size=4, per_video=8, seed=17)
    first = list(sampler)
    assert first == list(sampler)
    sampler.epoch = 2
    assert list(sampler) != first
    loader = torch.utils.data.DataLoader(
        Windows(store, 4),
        sampler=sampler,
        batch_size=3,
        num_workers=2,
        multiprocessing_context="spawn",
        collate_fn=collate,
    )
    batches = list(loader)
    assert [b["sample_count"] for b in batches] == [3, 3, 2]
    assert [w[:2] for batch in batches for w in batch["windows"]] == list(sampler)


def test_validation_covers_full_timeline_without_GT_in_model_inputs(monkeypatch):
    from ..graphs import Graph
    from . import validation
    from .data import sparse_targets

    store = FixtureStore()
    cfg = config()
    cfg.update(window_frames=6, validation_stride=4)

    class Model:
        def eval(self):
            return self

    def fixture_logits(model, batch, chunk_size, precision):
        assert "labels" not in batch and "supervision_mask" not in batch
        graph = Graph(
            "synthetic_m1",
            batch["nodes"].numpy(),
            batch["spatial_edges"].numpy(),
            batch["temporal_edges"].numpy(),
            batch["self_edges"].numpy(),
            batch["query_edges"].numpy(),
        )
        labels, _ = sparse_targets(graph, store.targets["synthetic_m1"], [])
        return np.where(labels, 20, -20).astype(np.float32)

    monkeypatch.setattr(validation, "predict_batch", fixture_logits)
    report = validation.validate(Model(), store, cfg, "cpu", batch_size=2)
    assert report["metrics"]["0.5"]["50"]["R"] == 100
    assert report["evaluated_video_ids"] == ["synthetic_m1"]


def test_runner_epoch_checkpoint_resume_and_completed_skip_on_CPU(tmp_path, monkeypatch):
    from ..contracts import read_json
    from . import runner

    cfg = small_config()
    cfg.update(
        output_root=str(tmp_path),
        window_frames=6,
        validation_stride=4,
        loader_workers=0,
        effective_batch_windows=4,
        windows_per_fit_video_per_epoch=6,
        max_epochs=2,
        min_epochs=2,
    )
    profile = {"models": {"mlp": {"batch_size": 2}}}
    monkeypatch.setattr(runner, "Store", lambda *args: FixtureStore())
    events = []

    class Tracker:
        url = "fixture://no-network"

        def __init__(self, interrupt):
            self.config, self.summary, self.interrupt = {}, {}, interrupt

        def log(self, metrics):
            events.append(metrics["epoch"])
            if self.interrupt:
                raise KeyboardInterrupt("simulated interruption after durable epoch checkpoint")

        def finish(self, exit_code):
            events.append(("finish", exit_code))

    monkeypatch.setattr(runner, "tracking", lambda *args: Tracker(True))
    out = tmp_path / "mlp/seed-17"
    try:
        with pytest.raises(KeyboardInterrupt):
            runner.train(cfg, "mlp", 17, profile, "fixture-cache", torch.device("cpu"))
        assert read_json(out / "history.json")[-1]["epoch"] == 1
        assert (out / "last.pt").is_file() and not (out / "complete.json").exists()
        monkeypatch.setattr(runner, "tracking", lambda *args: Tracker(False))
        runner.train(cfg, "mlp", 17, profile, "fixture-cache", torch.device("cpu"))
        assert [r["epoch"] for r in read_json(out / "history.json")] == [1, 2]
        assert read_json(out / "complete.json")["epochs"] == 2
        assert read_json(out / "history.json")[-1]["optimizer_steps"] == 2  # group 4 then group 2
        before = list(events)
        runner.train(cfg, "mlp", 17, profile, "fixture-cache", torch.device("cpu"))
        assert events == before
        assert not torch.cuda.is_initialized()
    finally:
        for path in tmp_path.rglob("*.pt"):
            path.unlink()


@pytest.mark.parametrize("kind", ["mlp", "static_gat", "st_gat"])
def test_tiny_fit_learns_positive_labels_and_deletes_test_weights(kind, tmp_path):
    cfg = config()
    result = overfit(kind, cfg, sample(0, 4), torch.device("cpu"), tmp_path, steps=200)
    write_json(tmp_path / "cpu_overfit.json", result)
    assert result["passed"], result
    assert not list(Path(tmp_path).rglob("*.pt"))
