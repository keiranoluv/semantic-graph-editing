"""Only a complete, hash-verified, sampling-matched three-seed matrix is reported."""

import json

import pytest

from src.datasets.gqa_full import digest
from src.knowledge.transe.core import sha256
from tools.run_gqa_kg_v2 import full_summary


def test_summary_checks_common_prefixes_and_reports_paired_deltas(tmp_path):
    experiment = {
        "config": {"seeds": [42, 123, 3407]},
        "long_tail": {"groups": {"head": [1], "body": [2], "tail": [3]}},
    }
    output, reference = tmp_path / "kg", tmp_path / "baseline"
    for index, seed in enumerate(experiment["config"]["seeds"]):
        metrics = {}
        for protocol, prefix in (("gc", ""), ("ng", "ng-")):
            metrics[protocol] = {prefix + name: 0.1 + index * 0.01 for name in ("R@50", "R@100", "mR@50", "mR@100")}
            metrics[protocol]["per_predicate"] = {
                str(i): {prefix + f"R@{k}": 0.1 + index * 0.01 for k in (50, 100)} for i in (1, 2, 3)
            }
        folder = output / "gat-kg" / str(seed)
        folder.mkdir(parents=True)
        (folder / "best.pt").write_bytes(b"synthetic checkpoint checksum fixture")
        result = {
            "identity": {"experiment_sha256": digest(experiment), "model": "gat-kg", "seed": seed},
            "status": "PASS",
            "checkpoint_sha256": sha256(folder / "best.pt"),
            "validation": metrics,
            "history": [{"sampling_trace_sha256": "same"}],
            "epochs_completed": 1,
            "selected_epoch": 1,
            "stopping_reason": "max_epochs",
        }
        (folder / "result.json").write_text(json.dumps(result))
        old = reference / "gat" / str(seed)
        old.mkdir(parents=True)
        (old / "result.json").write_text(
            json.dumps(
                {
                    "validation": metrics,
                    "history": [{"sampling_trace_sha256": "same"}, {"sampling_trace_sha256": "later"}],
                }
            )
        )
    summary = full_summary(experiment, output, reference)
    assert summary["status"] == "PASS" and summary["test_used"] is False
    assert summary["metrics"]["mR@50"]["mean"] == pytest.approx(0.11)
    assert summary["metrics"]["mR@50"]["std_sample"] == pytest.approx(0.01)
    assert summary["metrics"]["ng-mR@100"]["paired_deltas"] == [0, 0, 0]
    assert summary["long_tail"]["gat-kg"]["gc"]["tail"]["mR@50"] == pytest.approx([0.1, 0.11, 0.12])
    (output / "gat-kg/42/best.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checkpoint"):
        full_summary(experiment, output, reference)
