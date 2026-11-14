"""CPU-only lexical grounding, parity, learning, and resumable queue tests."""
from dataclasses import replace

import pytest
import torch
from test_magicbrush_experiment_v3 import fixture

from src.editing.operations import EditGraph
from tools.emnlp2020_graph_modification_v1.dataset import read, sha256, write
from tools.magicbrush_experiment_v3.model import StructuredPredictor as BasePredictor
from tools.magicbrush_experiment_v3.vocabulary import prototype_vectors
from tools.magicbrush_lexical_pilot_v1 import runner
from tools.magicbrush_lexical_pilot_v1.features import GroundingStore, collate_inputs, lexical, words
from tools.magicbrush_lexical_pilot_v1.model import StructuredPredictor
from tools.magicbrush_lexical_pilot_v1.training import train_one


@pytest.fixture(autouse=True)
def cpu_runtime():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def test_word_boundaries_plural_and_input_candidate_roles():
    labels = ("chair", "table", "car", "carpet", "dog")
    lf, nf, _, sub = lexical("Put chairs on the table", (("table", ()),), (), labels)
    assert lf[0][0] == lf[0][2] == lf[0][3] == 1
    assert lf[1][5] == 1 and lf[1][2] == 0
    assert sub == 0
    lf, _, _, _ = lexical("Add a carpet", (), (), labels)
    assert lf[2][0] == 0 and lf[3][0] == 1
    assert words("women with knives") == ("woman", "with", "knife")
    lf, _, _, sub = lexical("Replace the table with a chair", (("table", ()),), (), labels)
    assert sub == 1 and lf[0][4] == 1 and lf[1][4] == 0


def test_source_grounding_follows_nodes_and_attributes_not_slot_zero():
    nodes = (("table", ()), ("chair", ("red",)), ("chair", ("blue",)))
    _, nf, ef, _ = lexical("Remove the red chair", nodes, ((1, "beside", 0),), ("chair", "table"))
    assert nf[0][0] == 0 and nf[1][0] == nf[2][0] == 1
    assert nf[1][2] > nf[2][2] and nf[1][3] == 0
    assert ef[0][4] == 1 and ef[0][5] == 0


@pytest.mark.parametrize("kind", ["mlp", "gnn"])
def test_disabled_residuals_exactly_match_baseline_initialization_and_logits(kind):
    es, base, cfg, identity = fixture(all_kinds=True)
    store = GroundingStore(base, identity["vocabulary"])
    kwargs = dict(
        kind=kind,
        vocabulary=identity["vocabulary"],
        prototypes=prototype_vectors(store, identity["vocabulary"]),
        text_dim=512,
        **cfg["model"],
    )
    torch.manual_seed(42)
    old = BasePredictor(**kwargs).eval()
    torch.manual_seed(42)
    new = StructuredPredictor(**kwargs).eval()
    assert all(torch.equal(value, new.state_dict()[key]) for key, value in old.state_dict().items())
    new.enable_labels = False
    new.enable_ids = False
    batch = collate_inputs([es["train"][0].model_input()], store)
    with torch.inference_mode():
        a, b = old(batch), new(batch)
    assert all(torch.equal(value, b[key]) for key, value in a.items())
    with pytest.raises(ValueError, match="Only source"):
        collate_inputs([{**es["train"][0].model_input(), "target": es["train"][0].target}], store)
    changed = replace(es["train"][0], target=EditGraph(), operations=())
    second = collate_inputs([changed.model_input()], store)
    assert all(torch.equal(value, second[key]) for key, value in batch.items())


@pytest.mark.parametrize("kind", ["mlp", "gnn"])
def test_grounded_model_can_learn_and_resume_completed_cpu_example(tmp_path, kind):
    es, base, cfg, identity = fixture()
    cfg.update(epochs=60, minimum_epochs=10, patience=8, lr=0.01)
    store = GroundingStore(base, identity["vocabulary"])
    result = train_one(es, store, cfg, identity, kind, 42, tmp_path, device="cpu")
    assert result["validation"]["exact_target_graph_accuracy"] == 1
    assert train_one(es, store, cfg, identity, kind, 42, tmp_path, device="cpu") == result


def test_queue_resume_inference_ablations_and_tamper_detection(tmp_path, monkeypatch):
    examples, base, cfg, identity = fixture()
    cfg.update(
        seed=42,
        epochs=1,
        minimum_epochs=1,
        patience=1,
        output="runs/lexical",
        provenance={"baseline_summary": {"path": "baseline.json"}},
    )
    store = GroundingStore(base, identity["vocabulary"])
    gap = replace(examples["val"][0], id="gap", target=examples["val"][0].source, operations=())
    validation = examples["val"] + [gap]
    write(
        tmp_path / "baseline.json",
        {
            "models": {
                k: {"evaluation": {"constrained": {"micro": {"f1": 0.0}, "exact_target_graph_accuracy": 0.0}}}
                for k in cfg["models"]
            }
        },
    )
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    # Only emulate CUDA availability: all actual model calls explicitly use CPU.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    original_train, original_finish = runner.train_one, runner.finish
    calls = []
    interrupt = [True]

    def cpu_train(es, st, config, ident, kind, seed, out):
        if kind == "gnn" and interrupt[0]:
            interrupt[0] = False
            raise KeyboardInterrupt("Synthetic interruption")
        calls.append(kind)
        return original_train(es, st, config, ident, kind, seed, out, device="cpu")

    def cpu_finish(*args):
        return original_finish(*args, device="cpu")

    monkeypatch.setattr(runner, "train_one", cpu_train)
    monkeypatch.setattr(runner, "finish", cpu_finish)
    with pytest.raises(KeyboardInterrupt):
        runner.execute(cfg, examples, validation, store, identity, {})
    runner.execute(cfg, examples, validation, store, identity, {})
    out = tmp_path / cfg["output"]
    summary = read(out / "summary.json")
    assert calls == ["mlp", "gnn"] and summary["test_used"] is False
    for result in summary["models"].values():
        assert set(result["ablation"]["results"]) == {"without_labels", "without_ids", "without_both"}
        assert result["evaluation"]["schema_gap_pairs"] == 1
        assert result["evaluation"]["gap_diagnostics"][0]["correctness_assessed"] is False
    before = sha256(out / "summary.json")
    runner.execute(cfg, examples, validation, store, identity, {})
    assert calls == ["mlp", "gnn"] and sha256(out / "summary.json") == before
    (out / "mlp/42/ablation.json").write_text("{}")
    with pytest.raises(ValueError, match="modified"):
        runner.execute(cfg, examples, validation, store, identity, {})
