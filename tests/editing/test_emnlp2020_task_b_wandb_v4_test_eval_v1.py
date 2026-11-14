"""CPU checks for eval-only code; synthetic weights are never experiment results."""

import sys
from types import SimpleNamespace

import pytest
import torch

from src.editing.operations import EditGraph, Node, Operation, apply_operations
from tools import eval_emnlp2020_task_b_wandb_v4_test_v1 as evaluation
from tools.edit_model_v1.data import Example
from tools.edit_model_v1.text import TextStore, text_key
from tools.emnlp2020_graph_modification_v1.dataset import read, sha256, write
from tools.emnlp2020_task_b_gpu_v3.runner import inputs_tensor, make_model
from tools.emnlp2020_task_b_v2.model import ActionVocabulary


def test_eval_requires_training_audit_before_test_loading(tmp_path, monkeypatch):
    config = read("configs/level2/emnlp2020-task-b-wandb-v4.json")
    config["output"] = str(tmp_path / "missing-main-run")
    monkeypatch.setattr(evaluation, "load_bundle", lambda *a: pytest.fail("test must stay unopened"))
    with pytest.raises(FileNotFoundError, match="experiment.json"):
        evaluation.evaluate_test(config, tmp_path / "eval", device="cpu")
    with pytest.raises(ValueError, match="new_output"):
        evaluation.evaluate_test(config, config["output"], device="cpu")


def test_cpu_eval_six_frozen_models_no_optimizer_no_wandb_resume_and_tamper(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    config = read("configs/level2/emnlp2020-task-b-wandb-v4.json")
    config.update(
        output=str(tmp_path / "train"), device="cpu", hidden=8, max_nodes=1, max_edges=1, max_decode_tokens=12
    )
    source = EditGraph((Node("n001", "cat"),), ())
    ops = (Operation("replace_node", source.nodes[0], Node("n001", "dog")),)
    e = Example("test-fixture", "test", "g", "make dog", source, apply_operations(source, ops), ops)
    store = TextStore([text_key(t) for t in [e.instruction, '["cat", []]']], torch.randn(2, 512))
    vocabulary = ActionVocabulary.fit([e])
    tensors = inputs_tensor([e], store, config)
    runs = []
    for kind in config["models"]:
        for seed in config["seeds"]:
            path = tmp_path / "train" / f"{kind}-seed{seed}"
            path.mkdir(parents=True)
            model = make_model(config, kind, vocabulary)
            torch.save({"state_dict": model.state_dict()}, path / "best.pt")
            write(path / "result.json", {"checkpoint_sha256": sha256(path / "best.pt")})
            runs.append({"model": kind, "seed": seed})
    monkeypatch.setattr(
        evaluation, "prepare_inputs", lambda *a: (runs, [e], vocabulary, store, tensors, {"fixture": True})
    )
    monkeypatch.setattr(evaluation, "retrieval_baseline", lambda *a: {"fixture": True})
    monkeypatch.setattr(torch.optim, "AdamW", lambda *a, **kw: pytest.fail("eval must not create optimizer"))
    monkeypatch.setitem(
        sys.modules, "wandb", SimpleNamespace(init=lambda *a, **kw: pytest.fail("eval must not initialize W&B"))
    )
    output = tmp_path / "test-output"
    result = evaluation.evaluate_test(config, output, device="cpu")
    assert len(result["runs"]) == 6 and result["examples"] == 1
    assert result["WandB_initialized"] is False and result["training_performed"] is False
    assert all(r["metrics"]["examples"] == 1 for r in result["runs"])
    monkeypatch.setattr(evaluation, "make_model", lambda *a: pytest.fail("completed eval must be reused"))
    assert evaluation.evaluate_test(config, output, device="cpu") == result
    (tmp_path / "train/mlp-seed42/best.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint_changed_after_audit"):
        evaluation.evaluate_test(config, output, device="cpu")
