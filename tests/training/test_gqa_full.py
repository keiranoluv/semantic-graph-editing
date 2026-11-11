"""Resume/integrity tests use synthetic fixtures, never experiment evidence."""

import copy
import io
import zipfile

import numpy as np
import pytest
import torch
from PIL import Image

from src.datasets.gqa_full import GQAFullDataset, atomic_write, cohort_for, digest, prepare_cache, verify_cache
from src.datasets.schemas import SceneGraphRecord, SceneObject, SceneRelation
from src.graph.builder import SceneGraphBuilder
from src.knowledge.transe.core import sha256
from src.training.gqa_full import MODELS, train_run
from tools.run_gqa_full import exclusive, load_config


@pytest.fixture(autouse=True)
def runtime(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    old, threads = torch.are_deterministic_algorithms_enabled(), torch.get_num_threads()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    yield
    torch.use_deterministic_algorithms(old)
    torch.set_num_threads(threads)


def fixtures(device):
    torch.manual_seed(11)
    record = SceneGraphRecord(
        "a",
        100,
        100,
        (SceneObject(1, 1, (0, 0, 20, 20)), SceneObject(2, 2, (30, 0, 50, 20))),
        (SceneRelation(1, 2, 1), SceneRelation(1, 2, 3)),
    )
    graph = SceneGraphBuilder().build(record, torch.randn(2, 517))
    empty = SceneGraphBuilder().build(SceneGraphRecord("b", 100, 100), torch.empty(0, 517))
    experiment = {
        "config": {
            "scope": "fixture",
            "device": device,
            "epochs": 2,
            "lr": 0.001,
            "weight_decay": 0.01,
            "train_batch_size": 1,
            "val_batch_size": 1,
            "gradient_clip_norm": 5.0,
            "checkpoint_every_batches": 1,
        },
        "dimensions": {
            "geometry_dim": 517,
            "embedding_dim": 4,
            "hidden_dim": 8,
            "num_classes": 4,
            "num_object_classes": 4,
            "semantic_dim": 8,
            "pe_dropout": 0.2,
        },
    }
    vectors = [torch.randn(4, 300), torch.randn(4, 300)]
    return [graph, empty, graph], [graph], vectors, experiment


def assert_nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_nested_equal(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            assert_nested_equal(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("kind", MODELS)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mid_epoch_resume_exact_model_optimizer_rng_and_metrics(tmp_path, kind, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    train, val, vectors, experiment = fixtures(device)
    continuous, resumed = tmp_path / "continuous", tmp_path / "resumed"
    expected = train_run(kind, 42, train, val, vectors, experiment, continuous)
    assert train_run(kind, 42, train, val, vectors, experiment, resumed, stop_after_batches=1) is None
    assert not (resumed / "result.json").exists()
    # Arbitrary work between processes must not affect resumed RNG state.
    torch.manual_seed(999)
    torch.randn(30, device=device)
    actual = train_run(kind, 42, train, val, vectors, experiment, resumed)
    for key in ("history", "validation", "initial_state_sha256", "selected_epoch"):
        assert expected[key] == actual[key]
    left, right = [torch.load(p / "last.pt", map_location="cpu", weights_only=True) for p in (continuous, resumed)]
    for key in ("model", "optimizer", "torch_rng", "cuda_rng", "sampler_rng", "best_model", "total_batches"):
        assert_nested_equal(left[key], right[key])
    assert train_run(kind, 42, train, val, vectors, experiment, resumed) == actual
    changed = copy.deepcopy(experiment)
    changed["config"]["lr"] *= 2
    with pytest.raises(ValueError, match="Pinned identity"):
        train_run(kind, 42, train, val, vectors, changed, resumed)


def test_atomic_checkpoint_failure_keeps_previous_file(tmp_path, monkeypatch):
    train, val, vectors, experiment = fixtures("cpu")
    assert train_run("pe-plpr", 42, train, val, vectors, experiment, tmp_path, stop_after_batches=1) is None
    expected = sha256(tmp_path / "last.pt")

    def fail_save(value, file):
        file.write(b"incomplete")
        raise OSError("simulated disk failure")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(OSError, match="disk failure"):
        train_run("pe-plpr", 42, train, val, vectors, experiment, tmp_path, stop_after_batches=1)
    assert sha256(tmp_path / "last.pt") == expected


def test_empty_epoch_never_claims_success(tmp_path):
    train, val, vectors, experiment = fixtures("cpu")
    with pytest.raises(ValueError, match="No training pairs"):
        train_run("pe-plpr", 42, [train[1]], val, vectors, experiment, tmp_path)
    assert not (tmp_path / "result.json").exists()


def test_cache_resume_recovery_lazy_loading_and_corruption(tmp_path, monkeypatch):
    import clip

    from scripts import cache_vg150_regions

    root = tmp_path / "raw"
    (root / "sceneGraphs").mkdir(parents=True)
    (root / "archives").mkdir()
    vocabulary = {"label_to_idx": {"a": 1}, "predicate_to_idx": {"left": 1}, "attribute_to_idx": {}}
    obj = {"name": "a", "x": 0, "y": 0, "w": 2, "h": 2, "attributes": [], "relations": []}
    raw = {i: {"width": 4, "height": 4, "objects": {"1": obj}} for i in ["1", "2", "3"]}
    annotation = root / "sceneGraphs/train_sceneGraphs.json"
    atomic_write(annotation, raw)
    manifest = {
        "splits": {"train": ["1", "2"], "val": ["3"], "test": ["4"]},
        "vocabulary": vocabulary,
        "annotation_sha256": {"train_sceneGraphs.json": sha256(annotation)},
    }
    atomic_write(tmp_path / "manifest.json", manifest)
    (tmp_path / "weights").write_text("fixture only")
    image = io.BytesIO()
    Image.new("RGB", (4, 4)).save(image, format="JPEG")
    with zipfile.ZipFile(root / "archives/images.zip", "w") as z:
        for i in raw:
            z.writestr(f"{i}.jpg", image.getvalue())
    monkeypatch.setattr(clip, "load", lambda *a, **kw: (torch.nn.Linear(1, 1), None))
    monkeypatch.setattr(
        cache_vg150_regions,
        "extract_vectors",
        lambda model, transform, image, boxes, batch, device: np.ones((len(boxes), 512), dtype=np.float32)
        / np.sqrt(512),
    )
    cache = tmp_path / "cache"
    config = {
        "scope": "full",
        "manifest": str(tmp_path / "manifest.json"),
        "root": str(root),
        "cache": str(cache),
        "weights": str(tmp_path / "weights"),
        "cache_shard_size": 1,
        "clip_batch_size": 2,
        "device": "cpu",
    }
    assert not prepare_cache(config, stop_requested=lambda: (cache / "train-00000.h5.json").exists())
    first_hash = sha256(cache / "train-00000.h5")
    (cache / "train-00000.h5.json").unlink()  # crash after rename, before marker
    (cache / "train-00001.h5.tmp").write_bytes(b"torn HDF5 file")
    assert prepare_cache(config)
    assert sha256(cache / "train-00000.h5") == first_hash
    metadata = verify_cache(cache)
    assert metadata["identity"]["test_used"] is False
    dataset = GQAFullDataset(cache, metadata, "train")
    try:
        assert len(dataset) == 2
        assert dataset[0].x.shape == (1, 517)
        assert dataset[1].edge_index.shape == (2, 0)
        assert [g.image_id for g in dataset[:]] == ["1", "2"]
    finally:
        dataset.close()
    assert prepare_cache(config)  # completed no-op verifies hashes
    with (cache / "train-00000.h5").open("ab") as f:
        f.write(b"corruption")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_cache(cache)


def test_config_split_and_concurrent_writer_guards(tmp_path):
    config = load_config("configs/level2/gqa-full-v1.json")
    assert len(config["models"]) * len(config["seeds"]) == 15
    config["models"] = ["pe-plpr"]
    atomic_write(tmp_path / "invalid.json", config)
    with pytest.raises(ValueError, match="Full matrix"):
        load_config(tmp_path / "invalid.json")
    with pytest.raises(ValueError, match="Non-disjoint"):
        cohort_for({"scope": "full"}, {"splits": {"train": ["1"], "val": ["1"], "test": []}})
    with exclusive(tmp_path / "run"), pytest.raises(RuntimeError, match="Another process"):
        with exclusive(tmp_path / "run"):
            pass
    assert digest({"a": 1, "b": 2}) == digest({"b": 2, "a": 1})
