"""KG dispatch must preserve sampling, resume and frozen trainer functions."""

import copy

import pytest
import torch

from src.training import gqa_full
from tests.training.test_gqa_full import assert_nested_equal, fixtures
from tools import train_gqa_pilot as evaluator
from tools.run_gqa_full_prepared import prepare_supervision
from tools.run_gqa_kg_pilot import create_model, frozen_trainer_extension, verify_reference


@pytest.mark.parametrize("kind", ["gat-kg", "gat-kg-off"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_kg_real_dispatch_resume_is_exact(tmp_path, monkeypatch, kind, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(1)
    train, val, _, experiment = fixtures(device)
    for graph in train + val:
        prepare_supervision(graph, experiment["dimensions"]["num_classes"])
    features, valid = torch.randn(4, 8), torch.tensor([False, True, False, True])
    features[~valid] = 0
    continuous, resumed = tmp_path / "continuous", tmp_path / "resumed"
    originals = gqa_full.create_model, gqa_full.forward, gqa_full.evaluate, gqa_full.collate, evaluator.collate
    with frozen_trainer_extension(features, valid):
        expected = gqa_full.train_run(kind, 42, train, val, [], experiment, continuous)
        assert gqa_full.train_run(kind, 42, train, val, [], experiment, resumed, stop_after_batches=1) is None
        torch.manual_seed(999)
        torch.randn(10, device=device)
        actual = gqa_full.train_run(kind, 42, train, val, [], experiment, resumed)
    assert originals == (
        gqa_full.create_model,
        gqa_full.forward,
        gqa_full.evaluate,
        gqa_full.collate,
        evaluator.collate,
    )
    for key in ("history", "validation", "initial_state_sha256", "selected_epoch"):
        assert_nested_equal(expected[key], actual[key])
    states = [torch.load(p / "last.pt", map_location="cpu", weights_only=True) for p in (continuous, resumed)]
    for key in ("model", "optimizer", "torch_rng", "cuda_rng", "sampler_rng", "best_model"):
        assert_nested_equal(states[0][key], states[1][key])
    assert states[0]["model"]["kg_features"].device.type == "cpu"
    changed = copy.deepcopy(experiment)
    changed["features_hash"] = "different"
    with frozen_trainer_extension(features, valid), pytest.raises(ValueError, match="Pinned identity"):
        gqa_full.train_run(kind, 42, train, val, [], changed, resumed)


def test_kg_off_ignores_feature_values_and_kg_buffers_remain_frozen():
    train, _, _, experiment = fixtures("cpu")
    d = experiment["dimensions"]
    torch.manual_seed(1)
    features = torch.randn(4, 8)
    valid = torch.ones(4, dtype=torch.bool)
    models = []
    for values in (features, features * 10):
        torch.manual_seed(42)
        models.append(create_model("gat-kg-off", d, values, valid).eval())
    g = train[0]
    outputs = [m(g.x, g.edge_index, g.edge_label_index, g.object_labels) for m in models]
    assert_nested_equal(outputs[0], outputs[1])
    outputs[0].sum().backward()
    assert models[0].kg_features.grad is None
    assert "kg_features" not in dict(models[0].named_parameters())
    assert any(p.grad is not None for p in models[0].parameters())


def test_reference_verification_rejects_modified_sources(tmp_path):
    from src.knowledge.transe.core import sha256

    path = tmp_path / "frozen.py"
    path.write_text("original")
    reference = {"code_sha256": {str(path): sha256(path)}}
    verify_reference(reference)
    path.write_text("modified")
    with pytest.raises(ValueError, match="Frozen reference source changed"):
        verify_reference(reference)
