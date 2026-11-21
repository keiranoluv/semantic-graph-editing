"""CPU checks for topology controls, temporal isolation, gradients and recovery."""

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from ..contracts import read_json
from ..m3.common import seed_all, write_json
from ..m3.data import collate as collate_m3
from ..m3.engine import backward_microbatch, denominators
from ..m3.models import RelationModel as M3Model
from ..m3.probe import stress_window
from ..m3.test_m3 import FixtureStore, sample
from .common import config, identity, verify_parent
from .data import collate
from .models import RelationModel
from .preflight import check_profile
from .probe import overfit

MODELS = config()["models"]


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(2)


def small_config():
    cfg = config()
    cfg.update(hidden_dim=16, class_embedding_dim=8, heads=2, blocks=1, dropout=0.0, pair_chunk_size=13)
    return cfg


def test_inherits_frozen_M3_protocol_and_keeps_parent_intact():
    verify_parent()
    cfg = config()
    assert cfg["pos_weight"] == 1 and cfg["decoder_threshold"] == 0.5
    assert cfg["effective_batch_windows"] == 32 and not cfg["knowledge_graph"]
    for kind, expected in [("pvsg_conv_k5_l3", 13), ("mlp_conv_k3_l2", 5)]:
        assert RelationModel(kind, cfg).receptive_field == expected


def test_sequence_index_preserves_candidates_targets_and_splits_visibility_gaps():
    samples = [sample(0, 10), sample(0, 10)]  # Same video/IDs, distinct window instances.
    parent, batch = collate_m3(samples), collate(samples)
    for name, value in parent.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, batch[name])
    count = batch["query_edges"].shape[1]
    seen, seven_twelve_segments = [], []
    for width in batch["sequence_widths"]:
        for row in batch[f"sequence_indices_{width}"]:
            row = row[row != count]
            seen.extend(row.tolist())
            q = batch["query_batch"][row]
            pairs = batch["query_track_pairs"][row]
            frames = batch["query_frame_indices"][row]
            assert torch.all(q == q[0]) and torch.all(pairs == pairs[0])
            assert torch.all(torch.diff(frames) == 1)
            if q[0] == 0 and pairs[0].tolist() == [7, 12]:
                seven_twelve_segments.append(frames.tolist())
    assert sorted(seen) == list(range(count))
    assert torch.equal(torch.tensor(seen)[batch["sequence_inverse"]], torch.arange(count))
    assert sorted(seven_twelve_segments) == [[0, 1, 2, 3, 4], [7, 8, 9]]


@pytest.mark.parametrize("kind", ["st_no_spatial", "st_no_temporal", "st_no_edges"])
def test_edge_ablation_keeps_parameters_queries_and_matches_manual_edge_removal(kind):
    seed_all(17)
    model = RelationModel(kind, small_config()).eval()
    parent = M3Model("st_gat", small_config()).eval()
    parent.load_state_dict(model.state_dict(), strict=True)
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in parent.parameters())
    batch = collate([sample(0, 8)])
    unchanged = {key: value.clone() for key, value in batch.items() if isinstance(value, torch.Tensor)}
    manual = dict(batch)
    if kind in ("st_no_spatial", "st_no_edges"):
        manual["spatial_edges"] = batch["spatial_edges"][:, :0]
    if kind in ("st_no_temporal", "st_no_edges"):
        manual["temporal_edges"] = batch["temporal_edges"][:, :0]
        manual["temporal_delta_frames"] = batch["temporal_delta_frames"][:0]
    torch.testing.assert_close(model(batch), parent(manual), atol=0, rtol=0)
    for key, value in unchanged.items():
        assert torch.equal(value, batch[key])


@pytest.mark.parametrize("kind", MODELS)
def test_batch_isolation_and_loss_chunk_gradient_equivalence(kind):
    seed_all(17)
    model = RelationModel(kind, small_config()).eval()
    samples = [sample(0, 4), sample(2, 7), sample(5, 10)]
    batch = collate(samples)
    with torch.no_grad():
        reference_logits = torch.cat([model(collate([s])) for s in samples])
        torch.testing.assert_close(model(batch), reference_logits, atol=1e-6, rtol=1e-5)
    chunked, accumulated = copy.deepcopy(model), copy.deepcopy(model)
    logits = model(batch)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch["labels"], reduction="none")
    loss = ((loss * batch["supervision_mask"]).sum(1) / denominators(batch)[batch["query_batch"]]).sum() / 3
    loss.backward()
    backward_microbatch(chunked, batch, chunk_size=13, group_windows=3)
    for part in [samples[:2], samples[2:]]:
        backward_microbatch(accumulated, collate(part), chunk_size=11, group_windows=3)
    for (name, p), q, r in zip(model.named_parameters(), chunked.parameters(), accumulated.parameters(), strict=True):
        if p.grad is None:
            assert q.grad is None and r.grad is None, name
        else:
            torch.testing.assert_close(p.grad, q.grad, atol=1e-7, rtol=2e-5, msg=name)
            torch.testing.assert_close(p.grad, r.grad, atol=1e-7, rtol=2e-5, msg=name)


@pytest.mark.parametrize("kind,radius", [("pvsg_conv_k5_l3", 6), ("mlp_conv_k3_l2", 2)])
def test_conv_receptive_field_and_no_context_from_distant_frames(kind, radius):
    seed_all(17)
    model = RelationModel(kind, small_config()).eval()
    with torch.no_grad():
        for layer in model.convs:
            layer.weight.fill_(0.01)
            layer.bias.fill_(0.1)
    batch = collate([stress_window(15, objects=2)])
    batch["visual"].requires_grad_(True)
    selected = torch.nonzero(batch["query_frame_indices"] == 7)[0, 0]
    model.encode(batch)[selected].sum().backward()
    magnitude = batch["visual"].grad.abs().sum(1)
    frames = batch["nodes"][:, 0]
    assert magnitude[(frames - 7).abs() > radius].sum() == 0
    assert magnitude[frames == 7 + radius].sum() > 0
    assert magnitude[frames == 7 - radius].sum() > 0


@pytest.mark.parametrize("kind", ["pvsg_conv_k5_l3", "mlp_conv_k3_l2"])
def test_conv_cannot_bridge_visibility_gap_or_reverse_pair(kind):
    seed_all(17)
    model = RelationModel(kind, small_config()).eval()
    batch = collate([sample()])
    batch["visual"].requires_grad_(True)
    pairs = batch["query_track_pairs"]
    selected = torch.nonzero((pairs[:, 0] == 7) & (pairs[:, 1] == 12) & (batch["query_frame_indices"] == 4))[0, 0]
    model.encode(batch)[selected].sum().backward()
    magnitude = batch["visual"].grad.abs().sum(1)
    assert magnitude[batch["nodes"][:, 0] >= 7].sum() == 0
    # Directional pair geometry is an independent input; reverse-pair rows must not contribute.
    batch = collate([sample()])
    batch["pair_geometry"].requires_grad_(True)
    model.encode(batch)[selected].sum().backward()
    other_pairs = (pairs[:, 0] != 7) | (pairs[:, 1] != 12)
    assert batch["pair_geometry"].grad[other_pairs].abs().sum() == 0


def test_padded_conv_equals_running_each_true_length_segment_separately():
    seed_all(17)
    model = RelationModel("pvsg_conv_k5_l3", small_config()).eval()
    with torch.no_grad():
        for layer in model.convs:
            layer.bias.fill_(0.7)  # Would leak through padding without per-layer masking.
        batch = collate([sample()])
        nodes = M3Model.encode(model, batch)
        q = batch["query_edges"]
        s, o = nodes[q[0]], nodes[q[1]]
        pairs = model.head[1](model.head[0](torch.cat((s, o, s * o, s - o, batch["pair_geometry"]), dim=1)))
        expected = torch.zeros_like(pairs)
        for width in batch["sequence_widths"]:
            for row in batch[f"sequence_indices_{width}"]:
                row = row[row != len(pairs)]
                x = pairs[row].T.unsqueeze(0)
                for layer in model.convs:
                    x = torch.relu(layer(x))
                expected[row] = x.squeeze(0).T
        torch.testing.assert_close(model.encode(batch), expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("kind", MODELS)
def test_zero_query_window_is_valid(kind):
    batch = collate([stress_window(4, objects=1)])
    model = RelationModel(kind, small_config())
    assert model(batch).shape == (0, 57)
    assert np.all(backward_microbatch(model, batch, chunk_size=11, group_windows=1) == 0)


def test_profile_requires_selected_model_identity_memory_and_cleanup(tmp_path):
    cfg, kind = config(), "pvsg_conv_k5_l3"
    result = dict(
        state="passed",
        selected_model=kind,
        identity=identity(cfg, "fixture"),
        test_checkpoints_remaining=0,
        gpu={"reserved_memory_budget_bytes": 100},
        models={
            kind: dict(
                batch_size=2,
                overfit={"passed": True},
                candidates=[
                    dict(
                        batch_size=2,
                        state="passed",
                        heavy={"peak_reserved_bytes": 80},
                        mixed={"peak_reserved_bytes": 70},
                    )
                ],
            )
        },
    )
    path = tmp_path / "profile.json"
    write_json(path, result)
    assert check_profile(path, cfg, "fixture", kind) == result
    with pytest.raises(ValueError):
        check_profile(path, cfg, "fixture", "st_no_edges")
    for change in [dict(test_checkpoints_remaining=1), dict(state="failed_overfit"), dict(identity={})]:
        write_json(path, dict(result, **change))
        with pytest.raises(ValueError):
            check_profile(path, cfg, "fixture", kind)
    result["models"][kind]["candidates"][0]["heavy"]["peak_reserved_bytes"] = 101
    write_json(path, result)
    with pytest.raises(ValueError):
        check_profile(path, cfg, "fixture", kind)


def test_runner_resume_with_M4_collation_and_online_tracking_boundary(tmp_path, monkeypatch):
    from . import runner
    from .summarize import summarize

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
    kind = "pvsg_conv_k5_l3"
    profile = {"models": {kind: {"batch_size": 2}}}
    monkeypatch.setattr(runner, "Store", lambda *args: FixtureStore())
    events = []

    class Tracker:
        url = "fixture://no-network"

        def __init__(self, interrupt):
            self.config, self.summary, self.interrupt = {}, {}, interrupt

        def log(self, metrics):
            events.append(metrics["epoch"])
            if self.interrupt:
                raise KeyboardInterrupt("durable epoch interruption")

        def finish(self, exit_code):
            events.append(("finish", exit_code))

    out = tmp_path / kind / "seed-17"
    monkeypatch.setattr(runner, "tracking", lambda *args: Tracker(True))
    try:
        with pytest.raises(KeyboardInterrupt):
            runner.train(cfg, kind, 17, profile, "fixture-cache", torch.device("cpu"))
        assert (out / "last.pt").exists() and not (out / "complete.json").exists()
        monkeypatch.setattr(runner, "tracking", lambda *args: Tracker(False))
        runner.train(cfg, kind, 17, profile, "fixture-cache", torch.device("cpu"))
        assert [row["epoch"] for row in read_json(out / "history.json")] == [1, 2]
        assert read_json(out / "history.json")[-1]["optimizer_steps"] == 2
        before = list(events)
        runner.train(cfg, kind, 17, profile, "fixture-cache", torch.device("cpu"))
        assert events == before
        report = summarize(cfg, kind)
        assert report["missing_seeds"] == [42, 2026] and not report["complete_three_seeds"]
        assert report["aggregate"]["mR50"]["sample_std"] is None
        assert not torch.cuda.is_initialized()
    finally:
        for path in tmp_path.rglob("*.pt"):
            path.unlink()


def test_main_tracking_rejects_offline_before_training(monkeypatch):
    from ..m3.runner import tracking

    monkeypatch.setenv("WANDB_MODE", "offline")
    with pytest.raises(ValueError, match="online"):
        tracking(config(), "pvsg_conv_k5_l3", 17, Path("unused"), "fixture", {})


def test_M4_validation_never_gives_targets_to_model(monkeypatch):
    from ..graphs import Graph
    from ..m3.data import sparse_targets
    from . import validation

    cfg = config()
    cfg.update(window_frames=6, validation_stride=4)
    store = FixtureStore()

    class Model:
        def eval(self):
            return self

    def perfect_logits(model, batch, chunk_size, precision):
        assert "labels" not in batch and "supervision_mask" not in batch
        assert "sequence_inverse" in batch
        graph = Graph(
            "synthetic_m1",
            *(batch[key].numpy() for key in ("nodes", "spatial_edges", "temporal_edges", "self_edges", "query_edges")),
        )
        labels, _ = sparse_targets(graph, store.targets["synthetic_m1"], [])
        return np.where(labels, 20, -20).astype(np.float32)

    monkeypatch.setattr(validation, "predict_batch", perfect_logits)
    report = validation.validate(Model(), store, cfg, "cpu", 2)
    assert report["metrics"]["0.5"]["50"]["R"] == 100
    assert report["evaluated_video_ids"] == ["synthetic_m1"]


def test_M4_collation_works_with_spawn_workers():
    from ..m3.data import FitSampler, Windows

    store = FixtureStore()
    sampler = FitSampler(store, 4, 5, 17)
    loader = torch.utils.data.DataLoader(
        Windows(store, 4),
        sampler=sampler,
        batch_size=2,
        collate_fn=collate,
        num_workers=2,
        multiprocessing_context="spawn",
    )
    batches = list(loader)
    assert [b["sample_count"] for b in batches] == [2, 2, 1]
    assert [w[:2] for b in batches for w in b["windows"]] == list(sampler)
    assert all("sequence_inverse" in b for b in batches)


@pytest.mark.parametrize("kind", MODELS)
def test_full_width_positive_overfit_and_test_weights_cleanup(kind, tmp_path):
    result = overfit(kind, config(), sample(0, 4), torch.device("cpu"), tmp_path, steps=200)
    write_json(tmp_path / f"{kind}_cpu_overfit.json", result)
    assert result["passed"], result
    assert not list(tmp_path.rglob("*.pt"))
