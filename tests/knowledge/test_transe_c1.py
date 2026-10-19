"""Synthetic-only C.1 tests: no claim about real WordNet/ConceptNet scores."""

import csv
import json

import numpy as np
import pytest
import torch

from src.knowledge.transe.core import (
    TransE,
    check_gate,
    evaluate,
    load_kg,
    negative_sample,
    set_seed,
    split_triples,
    split_triples_transductive,
)


def make_kg(tmp_path, source="wordnet"):
    folder = tmp_path / "kg"
    folder.mkdir()
    prefix = "wn:" if source == "wordnet" else "cn:"
    (folder / "entities.json").write_text(json.dumps({"entities": [prefix + str(i) for i in range(12)]}))
    (folder / "relations.json").write_text(json.dumps({"relations": ["IsA"]}))
    with (folder / "triples.tsv").open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["subject_id", "relation_id", "object_id", "provenance"])
        w.writerows([(i, 0, (i + 1) % 12, "test") for i in range(12)])
    meta = {
        "schema_version": 1,
        "source": source,
        "ready_for_gat_kg_training": False,
        "triple_count": 12,
        "unique_triple_count": 12,
    }
    (folder / "metadata.json").write_text(json.dumps(meta))
    return folder


def test_kg_and_splits(tmp_path):
    kg = load_kg(make_kg(tmp_path))
    assert len(kg.entities) == 12 and len(kg.triples) == 12
    a = split_triples(kg.triples, 42)
    b = split_triples(kg.triples, 42)
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=False))
    assert not set(map(tuple, a[0])) & set(map(tuple, a[1]))
    assert len(a[0]) + len(a[1]) + len(a[2]) == 12


def test_negative_excludes_all_true(tmp_path):
    kg = load_kg(make_kg(tmp_path))
    known = set(map(tuple, kg.triples))
    neg = negative_sample(kg.triples, len(kg.entities), known, np.random.default_rng(17))
    assert not set(map(tuple, neg)) & known
    assert all(a[1] == b[1] for a, b in zip(kg.triples, neg, strict=False))


def test_reviewed_fails_closed(tmp_path):
    folder = make_kg(tmp_path)
    kg = load_kg(folder)
    check_gate("draft", kg, folder)
    with pytest.raises(ValueError, match="manifest"):
        check_gate("reviewed", kg, folder)
    manifest = tmp_path / "review.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "review_status": "human_approved",
                "kg_source": "wordnet",
                "kg_sha256": kg.hashes,
                "kg_path": str(folder.resolve()),
                "reviews": [],
            }
        )
    )
    with pytest.raises(ValueError, match="not training-ready"):
        check_gate("reviewed", kg, folder, manifest)


def test_transe_train_deterministic_and_metrics():
    set_seed(12)
    m = TransE(12, 2, dim=8)
    batch = torch.tensor([[0, 0, 1], [1, 0, 2]], dtype=torch.long)
    negatives = torch.tensor([[3, 0, 1], [1, 0, 4]], dtype=torch.long)
    optimizer = torch.optim.Adam(m.parameters(), lr=0.001)
    optimizer.zero_grad()
    loss = m.loss(batch, negatives)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    m.normalize_entities()
    assert torch.allclose(torch.linalg.vector_norm(m.entity.weight, dim=1), torch.ones(12), atol=1e-5)
    metrics = evaluate(m, np.array([[0, 0, 1]], dtype=np.int64), {(0, 0, 1)}, torch.device("cpu"))
    assert metrics["count"] == 2 and 0 < metrics["MRR"] <= 1


def test_reject_sameas_and_bad_ids(tmp_path):
    folder = make_kg(tmp_path)
    (folder / "relations.json").write_text(json.dumps({"relations": ["SameAs"]}))
    with pytest.raises(ValueError, match="SameAs"):
        load_kg(folder)
    (folder / "relations.json").write_text(json.dumps({"relations": ["IsA"]}))
    with (folder / "triples.tsv").open("a") as f:
        f.write("999\t0\t1\ttest\n")
    with pytest.raises(ValueError, match="outside vocabulary"):
        load_kg(folder)


def test_transductive_split_coverage_reproducible_and_disjoint():
    # Synthetic graph: leaf nodes, repeated relations, and disconnected components.
    triples = np.array(
        [(i, i % 2, i + 1) for i in range(30)] + [(i, 2, (i + 7) % 31) for i in range(31)], dtype=np.int64
    )
    a = split_triples_transductive(triples, seed=42)
    b = split_triples_transductive(triples, seed=42)
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=False))
    train, val, test = a
    assert len(train) + len(val) + len(test) == len(triples)
    sets = [set(map(tuple, part)) for part in a]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    e = set(train[:, 0]) | set(train[:, 2])
    r = set(train[:, 1])
    for part in (val, test):
        assert (set(part[:, 0]) | set(part[:, 2])).issubset(e)
        assert set(part[:, 1]).issubset(r)


def test_transductive_split_fails_when_holdout_infeasible():
    # Every edge has a uniquely occurring leaf -> all edges must remain in train.
    triples = np.array([(2 * i, 0, 2 * i + 1) for i in range(10)], dtype=np.int64)
    with pytest.raises(ValueError, match="infeasible"):
        split_triples_transductive(triples, seed=42)


def test_transductive_split_does_not_modify_input():
    triples = np.array([(i, 0, (i + 1) % 20) for i in range(20)], dtype=np.int64)
    before = triples.copy()
    split_triples_transductive(triples, seed=42)
    assert np.array_equal(triples, before)


def test_cli_bounded_evaluation_and_split_ratios(tmp_path, monkeypatch, capsys):
    from scripts import train_transe

    kg = make_kg(tmp_path)
    output = tmp_path / "bounded"
    args = [
        "train_transe",
        "--kg-dir",
        str(kg),
        "--output-dir",
        str(output),
        "--val-ratio",
        "0.2",
        "--test-ratio",
        "0.2",
        "--eval-max-triples",
        "1",
        "--dim",
        "8",
        "--epochs",
        "1",
        "--batch-size",
        "4",
    ]
    monkeypatch.setattr("sys.argv", args + ["--dry-run"])
    train_transe.main()
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["eval_subset"] is True
    assert manifest["train_count"] == 8 and manifest["val_count"] == 2 and manifest["test_count"] == 2
    assert not output.exists()
    monkeypatch.setattr("sys.argv", args)
    train_transe.main()
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["history"][0]["val"]["count"] == 2
    assert metrics["test"]["count"] == 2
    assert metrics["evaluation_protocol"]["val_total"] == 2
    assert metrics["evaluation_protocol"]["subset"] is True


def test_cli_invalid_ratios_and_eval_limit(tmp_path, monkeypatch):
    from scripts import train_transe

    kg = make_kg(tmp_path)
    for extra in (["--eval-max-triples", "0"], ["--val-ratio", "0.8", "--test-ratio", "0.3"], ["--val-ratio", "nan"]):
        monkeypatch.setattr(
            "sys.argv",
            ["train_transe", "--kg-dir", str(kg), "--output-dir", str(tmp_path / "invalid"), "--dry-run"] + extra,
        )
        with pytest.raises(SystemExit):
            train_transe.main()


def test_cli_eval_subset_deterministic(tmp_path, monkeypatch, capsys):
    from scripts import train_transe

    kg = make_kg(tmp_path)
    common = ["train_transe", "--kg-dir", str(kg), "--dry-run", "--eval-max-triples", "1"]
    for name in ("first", "second"):
        monkeypatch.setattr("sys.argv", common + ["--output-dir", str(tmp_path / name)])
        train_transe.main()
        obj = json.loads(capsys.readouterr().out)
        assert obj["eval_val_triples"] == obj["eval_test_triples"] == 1
