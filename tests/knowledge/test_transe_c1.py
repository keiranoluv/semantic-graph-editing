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
    sha256,
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


def test_negative_excludes_visible_true(tmp_path):
    kg = load_kg(make_kg(tmp_path))
    known = set(map(tuple, kg.triples))
    neg = negative_sample(kg.triples, len(kg.entities), known, np.random.default_rng(17))
    assert not set(map(tuple, neg)) & known
    assert all(a[1] == b[1] for a, b in zip(kg.triples, neg, strict=False))


def test_negative_filter_does_not_consult_holdout():
    positives = np.array([[0, 0, 1]], dtype=np.int64)
    train_known = {(0, 0, 1)}
    # Entity 2 is the only possible head corruption. Even if a held-out
    # positive uses it, train-time sampling cannot inspect that holdout.
    negative = negative_sample(positives, 3, train_known | {(0, 0, 2), (0, 0, 0), (1, 0, 1)}, np.random.default_rng(2))
    assert tuple(negative[0]) == (2, 0, 1)


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


def test_vectorized_filtered_ranking_matches_candidate_reference():
    set_seed(7)
    model = TransE(19, 2, dim=6)
    triples = np.array([[0, 0, 1], [3, 1, 4], [7, 0, 9]], dtype=np.int64)
    known = {(0, 0, 1), (2, 0, 1), (0, 0, 5), (3, 1, 4), (7, 0, 9), (8, 0, 9)}
    observed = evaluate(model, triples, known, torch.device("cpu"))
    ranks = []
    with torch.no_grad():
        for h, r, t in triples:
            for side in (0, 2):
                query = np.tile([h, r, t], (19, 1))
                query[:, side] = np.arange(19)
                scores = model.score(torch.as_tensor(query, dtype=torch.long)).tolist()
                gold = scores[h if side == 0 else t]
                for candidate in range(19):
                    triple = (candidate, int(r), int(t)) if side == 0 else (int(h), int(r), candidate)
                    if triple in known and triple != (int(h), int(r), int(t)):
                        scores[candidate] = float("inf")
                better = sum(score < gold - 1e-8 for score in scores)
                tied = sum(abs(score - gold) <= 1e-8 for score in scores)
                ranks.append(1 + better + (tied - 1) / 2)
    assert observed["count"] == len(ranks)
    assert observed["MRR"] == pytest.approx(np.mean([1 / rank for rank in ranks]))
    for k in (1, 3, 10):
        assert observed[f"Hits@{k}"] == pytest.approx(np.mean([rank <= k for rank in ranks]))


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
    assert manifest["negative_filter"] == "train_only"
    assert manifest["validation_rank_filter"] == "train_and_validation_only"
    assert manifest["train_count"] == 8 and manifest["val_count"] == 2 and manifest["test_count"] == 2
    assert not output.exists()
    monkeypatch.setattr("sys.argv", args)
    train_transe.main()
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["history"][0]["val"]["count"] == 2
    assert metrics["test"]["count"] == 2
    assert metrics["evaluation_protocol"]["val_total"] == 2
    assert metrics["evaluation_protocol"]["subset"] is True


def test_cli_filters_do_not_read_holdout_early(tmp_path, monkeypatch):
    from scripts import train_transe

    folder = make_kg(tmp_path)
    output = tmp_path / "filters"
    negative_sets = []
    evaluation_sets = []
    original_negative = train_transe.negative_sample
    original_evaluate = train_transe.evaluate

    def capture_negative(positives, n_entities, known, rng):
        negative_sets.append(set(known))
        return original_negative(positives, n_entities, known, rng)

    def capture_evaluate(model, triples, known, device):
        evaluation_sets.append(set(known))
        return original_evaluate(model, triples, known, device)

    monkeypatch.setattr(train_transe, "negative_sample", capture_negative)
    monkeypatch.setattr(train_transe, "evaluate", capture_evaluate)
    monkeypatch.setattr(
        "sys.argv",
        ["train_transe", "--kg-dir", str(folder), "--output-dir", str(output), "--epochs", "1", "--dim", "8"],
    )
    train_transe.main()
    with np.load(output / "split_indices.npz") as split:
        train = set(map(tuple, split["train"]))
        val = set(map(tuple, split["val"]))
        test = set(map(tuple, split["test"]))
    assert negative_sets and all(item == train for item in negative_sets)
    assert evaluation_sets == [train | val, train | val | test]


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


def test_cli_early_stop_tests_selected_best_checkpoint(tmp_path, monkeypatch):
    from scripts import train_transe

    folder = make_kg(tmp_path)
    output = tmp_path / "early_stop"
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_transe",
            "--kg-dir",
            str(folder),
            "--output-dir",
            str(output),
            "--dim",
            "8",
            "--epochs",
            "4",
            "--patience",
            "1",
            "--min-delta",
            "1",
            "--batch-size",
            "4",
        ],
    )
    train_transe.main()
    result = json.loads((output / "metrics.json").read_text())
    assert len(result["history"]) == 2
    assert result["selection"]["best_epoch"] == result["selection"]["selected_epoch"] == 1
    assert result["selection"]["early_stopped"] is True
    assert not (output / "epoch_0003.pt").exists()

    kg = load_kg(folder)
    best = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    assert best["epoch"] == 1
    model = TransE(len(kg.entities), len(kg.relations), dim=8)
    model.load_state_dict(best["model"])
    with np.load(output / "split_indices.npz") as split:
        expected_test = evaluate(model, split["test"], set(map(tuple, kg.triples)), torch.device("cpu"))
    assert result["test"] == expected_test


def test_official_batch_review_binds_selection_and_class_entities(tmp_path, monkeypatch):
    from scripts import prepare_level1_official_manifest

    folder = make_kg(tmp_path)
    selections = tmp_path / "selections.txt"
    selections.write_text("selected_synset\n" + "".join(f"{i % 12}\n" for i in range(1, 151)))
    overrides = tmp_path / "overrides.json"
    overrides.write_text(
        json.dumps(
            {
                "mappings": {f"class{i}": str(i % 12) for i in range(1, 151)},
                "review_provenance": {f"class{i}": {"class_id": i, "reviewer": "Toan"} for i in range(1, 151)},
            }
        )
    )
    mapping = tmp_path / "mapping.json"
    mapping.write_text(
        json.dumps(
            {
                "overrides_sha256": sha256(overrides),
                "rows": [{"class_id": 0, "label": "__background__", "status": "background", "synset_id": None}]
                + [
                    {"class_id": i, "label": f"class{i}", "status": "linked_manual", "synset_id": str(i % 12)}
                    for i in range(1, 151)
                ],
            }
        )
    )
    approval = tmp_path / "approval.json"
    approval.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "reviewer": "Toan",
                "review_date": "2026-10-25",
                "approval_scope": "all_150_representative_synsets",
                "statement": "Toan approved this batch",
                "per_class_rationale_provided": False,
            }
        )
    )
    (folder / "class_entity_ids.json").write_text(
        json.dumps({"class_entity_ids": [None] + [i % 12 for i in range(1, 151)]})
    )
    meta = json.loads((folder / "metadata.json").read_text())
    meta.update(
        ready_for_gat_kg_training=True,
        resolved_class_count=150,
        unresolved_class_ids=[],
        unreviewed_class_ids=[],
        input_mapping_sha256=sha256(mapping),
    )
    (folder / "metadata.json").write_text(json.dumps(meta))
    manifest = tmp_path / "review_manifest.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_level1_official_manifest",
            "--kg-dir",
            str(folder),
            "--mapping",
            str(mapping),
            "--overrides",
            str(overrides),
            "--selections",
            str(selections),
            "--approval",
            str(approval),
            "--output",
            str(manifest),
        ],
    )
    prepare_level1_official_manifest.main()
    kg = load_kg(folder)
    check_gate("reviewed", kg, folder, manifest)
    selections.write_text(selections.read_text() + "tampered\n")
    with pytest.raises(ValueError, match="selections path/hash mismatch"):
        check_gate("reviewed", kg, folder, manifest)
