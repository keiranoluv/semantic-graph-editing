from unittest.mock import patch

import pytest

from tools import level1_loss_fusion_test27_v1 as launcher


def test_plan_scope_and_no_overlap():
    plan = launcher.build_plan()
    entries = plan["entries"]
    assert len(entries) == 27
    assert sum(e["condition"].startswith("loss_") for e in entries) == 12
    assert sum(e["condition"].startswith("fusion_") for e in entries) == 15
    assert not any("relation_" in e["condition"] for e in entries)
    assert len({e["checkpoint_sha256"] for e in entries}) == 27


def test_reject_changed_plan():
    with patch.object(launcher.runner, "sha256", return_value="changed"):
        with pytest.raises(ValueError, match="Frozen source plan changed"):
            launcher.build_plan()
