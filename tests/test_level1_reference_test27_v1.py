import json
from unittest.mock import patch

import pytest

from tools import level1_reference_test27_v1 as runner


def test_resume_skips_all_inference_and_is_idempotent(tmp_path):
    entries = [
        dict(condition=f"c{i}", seed=s, checkpoint="cp", checkpoint_sha256="h")
        for i in range(9)
        for s in (42, 123, 3407)
    ]
    for entry in entries:
        row = dict(entry, **{k: 0.2 for k in runner.METRICS})
        (tmp_path / f"{entry['condition']}-seed{entry['seed']}-test26446.json").write_text(json.dumps(row))
    with patch.object(runner, "ROOT", tmp_path), patch.object(runner, "OUTPUT", tmp_path), patch.object(
        runner, "sha256", return_value="h"
    ), patch.object(runner, "validate"), patch.object(runner.subprocess, "run") as run:
        runner.execute({"entries": entries}, 128, False)
        runner.execute({"entries": entries}, 128, False)
        run.assert_not_called()


def test_rejects_legacy_report():
    with pytest.raises(ValueError, match="incompatible"):
        runner.validate(
            {"protocol": "PredCls_graph_constrained_top1_per_pair"},
            {"checkpoint_sha256": "h", "seed": 42},
            {"data_sha256": {}},
        )


def test_changed_checkpoint_stops_before_inference():
    entries = [
        dict(condition=f"c{i}", seed=s, checkpoint="cp", checkpoint_sha256="h")
        for i in range(9)
        for s in (42, 123, 3407)
    ]
    with patch.object(runner, "sha256", return_value="changed"), patch.object(runner.subprocess, "run") as run:
        with pytest.raises(ValueError, match="Checkpoint changed"):
            runner.execute({"entries": entries}, 128, False)
        run.assert_not_called()
