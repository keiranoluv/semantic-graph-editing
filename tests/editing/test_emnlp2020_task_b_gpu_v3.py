"""Device transfers and complete sealed protocol; no real-data metric fixtures."""

from dataclasses import replace

import pytest
import torch

from src.editing.operations import Edge, EditGraph, Node, Operation, apply_operations
from tools.edit_model_v1.data import Example
from tools.edit_model_v1.text import TextStore, text_key


def example():
    source = EditGraph((Node("n001", "café"), Node("n002", "dog")), (Edge("e001", "n001", "near", "n002"),))
    operations = (Operation("replace_node", source.nodes[0], Node("n001", 'blue "car"')),)
    return Example("x", "train", "group", "make it blue", source, apply_operations(source, operations), operations)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_device_protocol_six_runs_freeze_resume_tamper(tmp_path, monkeypatch, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; CPU contract is still tested")
    from tools.emnlp2020_graph_modification_v1.dataset import read, write
    from tools.emnlp2020_task_b_gpu_v3 import runner

    torch.set_num_threads(1)
    e = example()
    trainval = {"train": [e], "val": [replace(e, id="val", split="val", group="val-group")]}
    test = replace(e, id="test", split="test", group="test-group")
    texts = [e.instruction, '["café", []]', '["dog", []]', "near"]
    store = TextStore([text_key(t) for t in texts], torch.randn(len(texts), 512))
    dataset = tmp_path / "dataset"
    write(dataset / "trainval-receipt.json", {"fixture": True})
    write(dataset / "test-receipt.json", {"fixture": True})
    write(dataset / "clip-trainval/manifest.json", {"fixture": True})
    config = {
        "device": device,
        "dataset": str(dataset),
        "output": str(tmp_path / "run"),
        "weights": "unused",
        "models": ["mlp", "gnn"],
        "seeds": [42, 43, 44],
        "hidden": 8,
        "layers": 1,
        "max_nodes": 2,
        "max_edges": 1,
        "text_batch_size": 2,
        "batch_size": 2,
        "epochs": 1,
        "max_decode_tokens": 16,
        "lr": 0.001,
        "weight_decay": 0.01,
    }
    monkeypatch.setattr(runner, "load_bundle", lambda output, name: {"test": [test]} if name == "test" else trainval)
    monkeypatch.setattr(runner, "frozen_features", lambda *args: store)
    accessed = []

    def prepare_test(*args, **kwargs):
        freeze = read(tmp_path / "run/frozen-checkpoints.json")
        assert len(freeze["runs"]) == 6
        accessed.append("test")
        return {"test": {"usable": 1}}

    monkeypatch.setattr(runner, "prepare", prepare_test)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    freeze = runner.train(config)
    assert not accessed
    assert len(freeze["runs"]) == 6
    assert runner.train(config) == freeze
    result = runner.final_test(config, "unused")
    assert accessed == ["test"]
    assert len(result["runs"]) == 6
    assert result["runs"][0]["metrics"]["examples"] == 1
    assert runner.final_test(config, "unused") == result
    checkpoint = tmp_path / "run/mlp-seed42/best.pt"
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="frozen_checkpoint_changed"):
        runner.final_test(config, "archive-must-not-be-read")
    assert accessed == ["test", "test"]
