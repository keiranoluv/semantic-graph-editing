"""Selected-checkpoint metrics must not be confused with the final W&B epoch."""

import pytest

from tools.summarize_emnlp2020_task_b_wandb_v4_v1 import check_stopping, check_wandb_summary


def test_wandb_selected_metrics_are_used_even_when_last_epoch_f1_is_higher():
    result = {
        "best_epoch": 6,
        "epochs_completed": 9,
        "best_val_loss": 0.35,
        "validation": {"micro": {"f1": 0.8}},
        "stopping_reason": "early_stopping",
    }
    summary = {
        "best_epoch": 6,
        "epochs_completed": 9,
        "best_val_loss": 0.35,
        "selected_val/micro/f1": 0.8,
        "val/micro/f1": 0.9,
        "stopping_reason": "early_stopping",
    }
    check_wandb_summary(result, summary)
    summary["selected_val/micro/f1"] = 0.9
    with pytest.raises(ValueError, match="selected_summary_mismatch"):
        check_wandb_summary(result, summary)
    del summary["selected_val/micro/f1"]
    with pytest.raises(ValueError, match="selected_summary_mismatch"):
        check_wandb_summary(result, summary)


def test_stopping_audit_rejects_incomplete_or_changed_best_epoch():
    config = {"epochs": 20, "minimum_epochs": 1, "patience": 1, "min_delta": 0.01}
    result = {
        "epochs_completed": 2,
        "best_epoch": 1,
        "best_val_loss": 0.8,
        "stopping_reason": "early_stopping",
        "history": [
            {"epoch": 1, "val/loss": 0.8, "early_stop/bad_epochs": 0, "early_stop/best_epoch": 1},
            {"epoch": 2, "val/loss": 0.81, "early_stop/bad_epochs": 1, "early_stop/best_epoch": 1},
        ],
    }
    assert check_stopping(result, config)["best_epoch"] == 1
    with pytest.raises(ValueError, match="wrong_selected_epoch"):
        check_stopping({**result, "best_epoch": 2}, config)
    with pytest.raises(ValueError, match="incomplete_epoch_history"):
        check_stopping({**result, "epochs_completed": 3}, config)
