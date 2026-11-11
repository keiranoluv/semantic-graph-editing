"""Checkpoint continuity and compatibility tests."""

import pytest
import torch

from scripts.pilot_mlp import check_resume_compatibility, restore_training_state


def test_resume_config_mismatch_rejected():
    saved = {"model": "mlp", "batch_size": 512, "seed": 42, "lr": 1e-3}
    changed = {**saved, "batch_size": 2048}
    with pytest.raises(ValueError, match="batch_size"):
        check_resume_compatibility(saved, changed)


def test_resume_restores_optimizer_and_rng():
    model = torch.nn.Linear(2, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(42)
    loss = model(torch.ones(1, 2)).sum()
    loss.backward()
    opt.step()
    checkpoint = {
        "epoch": 3,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "sampler_rng": generator.get_state(),
        "torch_rng": torch.get_rng_state(),
        "python_rng": __import__("random").getstate(),
        "best_score": 0.15,
        "best_epoch": 2,
        "bad_epochs": 1,
        "history": [{"epoch": 1}, {"epoch": 2}, {"epoch": 3}],
        "config": {"model": "mlp"},
    }
    restored_model = torch.nn.Linear(2, 2)
    restored_opt = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    restored_generator = torch.Generator().manual_seed(19)
    state = restore_training_state(checkpoint, restored_model, restored_opt, restored_generator, torch.device("cpu"))
    assert state == (4, 0.15, 2, 1, checkpoint["history"])
    for expected, actual in zip(model.parameters(), restored_model.parameters(), strict=False):
        torch.testing.assert_close(expected, actual)
    assert restored_generator.get_state().equal(generator.get_state())
    assert restored_opt.state_dict()["state"]
