"""CPU checks of compact adapters and the cache-to-two-model launcher."""
from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from test_magicbrush_experiment_v3 import fixture
from test_magicbrush_overfit_v1 import probe_fixture

from tools.edit_model_v1.text import load_store
from tools.emnlp2020_graph_modification_v1.dataset import read, sha256, write
from tools.magicbrush_compact_overfit_v1 import runner
from tools.magicbrush_compact_v1.build import example_dict
from tools.magicbrush_compact_v1.projection import project_example
from tools.prepare_magicbrush import digest


@pytest.fixture(autouse=True)
def cpu_only():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def test_adapter_uses_train_only_and_rejects_modified_labels():
    es, _, _, _ = fixture(all_kinds=True)
    policy = read("configs/level2/magicbrush-compact-v1-policy.json")
    old = es["train"][0]
    new, audit = project_example(old, policy)
    row = {**example_dict(new), "original_label_sha256": digest(example_dict(old)), "flags": audit["flags"]}
    payload = {"records": [row, {"id": "val", "split": "val", "target_graph": "not decoded by train adapter"}]}
    loaded = runner.load_train(payload, [old], policy)
    assert len(loaded) == 1 and loaded[0].id == old.id
    broken = deepcopy(payload)
    broken["records"][0]["instruction"] = "Modified"
    with pytest.raises(ValueError, match="differs"):
        runner.load_train(broken, [old], policy)
    with pytest.raises(ValueError, match="Duplicate"):
        runner.load_train({"records": [row, row]}, [old], policy)


def test_previous_cohort_order_is_preserved_and_not_reselected():
    es, _, _, _ = fixture(all_kinds=True)
    train = [replace(es["train"][0], id=f"toy:{i}", group=f"group:{i}") for i in range(18)]
    receipt = {"cases": [{"id": e.id} for e in reversed(train[:16])]}
    assert [e.id for e in runner.select_fixed(train, receipt)] == [r["id"] for r in receipt["cases"]]
    train[0] = replace(train[0], split="val")
    with pytest.raises(ValueError, match="train-only"):
        runner.select_fixed(train, receipt)


def test_identical_tensor_inputs_are_rejected():
    es, store, _, _ = fixture()
    assert len(runner.verify_features(es["train"], store)) == 1
    with pytest.raises(ValueError, match="collide"):
        runner.verify_features(es["train"] * 2, store)


def test_cache_then_real_cpu_models_resume_and_verify_completed(tmp_path, monkeypatch):
    cohort, store, config, identity = probe_fixture()
    config.update(output="runs/level2/magicbrush-compact-overfit16-v1", text_cache="data/compact-cache", max_steps=100)
    identity = {**identity, "scope": "synthetic compact CPU test"}
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    # Emulate availability only. Cache creation and every training call below
    # use CPU tensors; this test never invokes CLIP or CUDA kernels.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    generations = []

    def fake_cache(es, cfg, ident, path, root, *, scope):
        assert [e.id for e in es] == [e.id for e in cohort] and scope == "trainval"
        if not (path / "manifest.json").exists():
            path.mkdir(parents=True)
            torch.save(store.vectors, path / "vectors.pt")
            write(
                path / "manifest.json",
                {
                    "identity": ident,
                    "vectors_sha256": sha256(path / "vectors.pt"),
                    "shape": list(store.vectors.shape),
                    "text_sha256": list(store.keys),
                },
            )
            generations.append("CPU cache created")
        return load_store(path, ident)

    monkeypatch.setattr(runner, "build_cache", fake_cache)
    real = runner.train_one
    calls = []
    interrupt = [True]

    def cpu_train(es, st, cfg, ident, kind, path):
        if kind == "gnn" and interrupt[0]:
            interrupt[0] = False
            raise KeyboardInterrupt("Synthetic interruption before GNN")
        if not (path / "completed-receipt.json").exists():
            calls.append(kind)
        return real(es, st, cfg, ident, kind, path, device="cpu")

    monkeypatch.setattr(runner, "train_one", cpu_train)
    report = {"comparison_note": "synthetic CPU test"}
    with pytest.raises(KeyboardInterrupt):
        runner.execute(config, cohort, identity, report)
    runner.execute(config, cohort, identity, report)
    out = tmp_path / config["output"]
    summary = read(out / "summary.json")
    assert calls == ["mlp", "gnn"] and len(generations) == 1
    assert all(r["status"] == "MEMORIZED" for r in summary["models"].values())
    assert summary["automatic_followup_training"] is False
    checksum = sha256(out / "summary.json")
    runner.execute(config, cohort, identity, report)
    assert calls == ["mlp", "gnn"] and sha256(out / "summary.json") == checksum
    (out / "mlp/42/predictions-train.json").write_text("[]")
    with pytest.raises(ValueError, match="modified"):
        runner.execute(config, cohort, identity, report)
