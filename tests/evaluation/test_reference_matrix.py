"""Freeze exactly the eligible matrices; exclude the audited relation-prior runs."""

import json
import sys

import pytest

from src.knowledge.transe.core import sha256
from tools.level1_reference_matrix import main


def prepare(tmp_path, family):
    conditions = (
        ["ce_off", "ce_on", "adjusted_off", "adjusted_on"]
        if family == "pair"
        else ["baseline", "pair_real", "pair_shuffled", "visual_real", "visual_shuffled"]
    )
    prefix = "clip-pair-kg-v1" if family == "pair" else "clip-fusion-v1"
    reports = tmp_path / "reports/level1"
    reports.mkdir(parents=True)
    cp = tmp_path / "checkpoint.pt"
    cp.write_bytes(b"frozen")
    for condition in conditions:
        for seed in (42, 123, 3407):
            report = {
                "checkpoint": str(cp),
                "checkpoint_sha256": sha256(cp),
                "data_sha256": {"data": "hash"},
                "config": {"visual_cache_manifest_sha256": "cache"},
            }
            (reports / f"{prefix}-{condition}-seed{seed}-val5000.json").write_text(json.dumps(report))
    return cp


@pytest.mark.parametrize("family,count", [("pair", 12), ("fusion", 15)])
@pytest.mark.parametrize("protocol", ["reference-v1", "reference-ng-v1"])
def test_dry_run_exact_scope(tmp_path, monkeypatch, capsys, family, count, protocol):
    prepare(tmp_path, family)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["matrix", "--matrix", family, "--protocol", protocol, "--dry-run"])
    main()
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == count
    assert all(f"--protocol {protocol}" in line for line in lines)
    assert all("relation_real" not in line and "relation_shuffled" not in line for line in lines)
    assert not (tmp_path / f"reports/level1/{protocol}-{family}").exists()


def test_changed_checkpoint_rejected(tmp_path, monkeypatch):
    cp = prepare(tmp_path, "pair")
    cp.write_bytes(b"changed")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["matrix", "--matrix", "pair", "--dry-run"])
    with pytest.raises(ValueError, match="checkpoint changed"):
        main()
