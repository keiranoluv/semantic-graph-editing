"""Synthetic CPU checks for schema coverage, full-split reporting and pilot resume."""
from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from test_magicbrush_experiment_v3 import fixture

from tools.edit_model_v1.codec import encode_actions
from tools.edit_model_v1.text import load_store
from tools.emnlp2020_graph_modification_v1.dataset import read, sha256, write
from tools.magicbrush_compact_pilot_v1 import data, runner
from tools.magicbrush_compact_v1.build import example_dict
from tools.magicbrush_compact_v1.projection import project_example
from tools.prepare_magicbrush import digest


@pytest.fixture(autouse=True)
def cpu_runtime():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def label_row(e):
    policy = read("configs/level2/magicbrush-compact-v1-policy.json")
    n, a = project_example(e, policy)
    return {**example_dict(n), "original_label_sha256": digest(example_dict(e)), "flags": a["flags"]}, policy


def test_validation_loader_preserves_gaps_and_rejects_payload_or_membership_change():
    es, _, _, _ = fixture()
    e = es["val"][0]
    gap = replace(e, id="gap", target=e.source, operations=())
    row, policy = label_row(e)
    empty, _ = label_row(gap)
    payload = {"records": [row, empty]}
    loaded = data.load_validation(payload, [e, gap], policy)
    assert [x.id for x in loaded] == [e.id, gap.id]
    assert [len(xs) for xs in data.partition(loaded)] == [1, 1]
    bad = deepcopy(payload)
    bad["records"][0]["instruction"] = "changed"
    with pytest.raises(ValueError, match="projection"):
        data.load_validation(bad, [e, gap], policy)
    with pytest.raises(ValueError, match="membership"):
        data.load_validation({"records": [row, row]}, [e, gap], policy)


def test_schema_gaps_never_get_exact_success_credit():
    es, _, _, _ = fixture()
    e = es["val"][0]
    gap = replace(e, id="gap", target=e.source, operations=())

    def row(ex):
        tokens = encode_actions(ex.operations)[1:]
        return {
            "id": ex.id,
            "tokens": tokens,
            "raw_tokens": tokens,
            "error": None,
            "raw_error": None,
            "operation_count": len(ex.operations),
        }

    report = data.evaluation_report([e, gap], [row(e), row(gap)])
    assert report["constrained"]["micro"]["f1"] == 1
    assert report["constrained"]["examples"] == 1
    assert report["schema_coverage"] == 0.5
    assert report["conservative_exact_graph_over_all_pairs"] == 0.5
    assert report["gap_diagnostics"][0]["correctness_assessed"] is False
    assert report["gap_diagnostics"][0]["predicted_operation_count"] == 0
    with pytest.raises(ValueError, match="IDs/order"):
        data.evaluation_report([e, gap], [row(gap), row(e)])


def test_pilot_runs_both_models_after_failed_gate_resumes_and_seals_all_validation(tmp_path, monkeypatch):
    examples, store, cfg, identity = fixture()
    cfg.update(
        seed=42,
        output="runs/level2/magicbrush-compact-pilot-v1",
        text_cache="cache",
        epochs=1,
        minimum_epochs=1,
        patience=1,
    )
    cfg["quality_gate"] = {"minimum_valid_transaction_rate": 1.0, "minimum_micro_f1": 1.0}
    val = examples["val"][0]
    gap = replace(val, id="gap", target=val.source, operations=())
    validation = [val, gap]
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    # CUDA availability is emulated; cache and every model call use CPU below.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    generations = []

    def fake_cache(es, cfg, ident, path, root, *, scope):
        assert [e.id for e in es] == [e.id for e in examples["train"] + validation]
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
            generations.append(1)
        return load_store(path, ident)

    monkeypatch.setattr(runner, "build_cache", fake_cache)
    original_train, original_finish = runner.train_one, runner.finish_validation
    calls = []
    interrupt = [True]

    def train(es, st, conf, ident, kind, seed, out):
        assert len(es["val"]) == 1 and es["val"][0].id == val.id
        if kind == "gnn" and interrupt[0]:
            interrupt[0] = False
            raise KeyboardInterrupt("Synthetic queue interruption")
        calls.append(kind)
        return original_train(es, st, conf, ident, kind, seed, out, device="cpu")

    def finish(*args, **kwargs):
        return original_finish(*args, **kwargs, device="cpu")

    monkeypatch.setattr(runner, "train_one", train)
    monkeypatch.setattr(runner, "finish_validation", finish)
    with pytest.raises(KeyboardInterrupt):
        runner.execute(cfg, examples, validation, identity, {})
    runner.execute(cfg, examples, validation, identity, {})
    out = tmp_path / cfg["output"]
    summary = read(out / "summary.json")
    assert calls == ["mlp", "gnn"] and len(generations) == 1
    assert all(not x["quality_gate"]["passed"] for x in summary["models"].values())
    assert summary["test_used"] is False and summary["automatic_followup_training"] is False
    assert all(x["evaluation"]["schema_gap_pairs"] == 1 for x in summary["models"].values())
    for kind in cfg["models"]:
        assert [r["id"] for r in read(out / kind / "42/predictions-val-all.json")] == [val.id, "gap"]
    checksum = sha256(out / "summary.json")
    runner.execute(cfg, examples, validation, identity, {})
    assert calls == ["mlp", "gnn"] and sha256(out / "summary.json") == checksum
    (out / "mlp/42/validation-report.json").write_text("{}")
    with pytest.raises(ValueError, match="modified"):
        runner.execute(cfg, examples, validation, identity, {})
