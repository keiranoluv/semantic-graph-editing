"""Synthetic-only C.1 tests: no claim about real WordNet/ConceptNet scores."""

import csv
import json

import numpy as np
import pytest
import torch

from src.knowledge.transe.core import TransE, check_gate, evaluate, load_kg, negative_sample, set_seed, split_triples


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
