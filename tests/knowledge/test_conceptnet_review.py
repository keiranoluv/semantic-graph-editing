"""Review workflow tests use invented ConceptNet assertions only."""

import csv
import hashlib
import io
import json

import pytest
import torch

from scripts.build_conceptnet_graph import write_outputs
from scripts.prepare_conceptnet_review import check, compile_reviewed, prepare
from src.knowledge.conceptnet_builder import extract_assertions
from src.knowledge.export import export_conceptnet, load_class_features
from src.knowledge.transe.core import TransE, check_gate, load_kg, sha256


def _fixture(tmp_path):
    rows = [{"class_id": 0, "label": "__background__", "status": "background"}]
    for class_id in range(1, 151):
        rows.append(
            {
                "class_id": class_id,
                "label": {1: "dog", 2: "bike"}.get(class_id, f"class {class_id}"),
                "status": "ambiguous" if class_id == 2 else "linked_auto",
            }
        )
    mapping = {"schema_version": 1, "source": "wordnet", "rows": rows}
    assertion = (
        "/a/dog\t/r/IsA\t/c/en/dog\t/c/en/animal\t"
        '{"dataset":"/d/omcs","license":"cc:by-sa/4.0","weight":2.0}\n'
        "/a/bike\t/r/IsA\t/c/en/bike/n\t/c/en/vehicle\t"
        '{"dataset":"/d/omcs","license":"cc:by-sa/4.0","weight":2.0}\n'
    )
    source = tmp_path / "synthetic.tsv"
    source.write_text(assertion)
    kg_dir = tmp_path / "kg"
    outputs, _ = extract_assertions(mapping, source, expected_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    write_outputs(kg_dir, outputs)
    review_dir = tmp_path / "review"
    review_outputs, _ = prepare(kg_dir)
    write_outputs(review_dir, review_outputs)
    return kg_dir, review_dir


def _update_rows(path, changes):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for class_id, values in changes.items():
        rows[class_id - 1].update(values)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(stream.getvalue())


def test_prepare_is_blank_and_review_is_checked(tmp_path):
    kg_dir, review_dir = _fixture(tmp_path)
    original = (review_dir / "review.csv").read_bytes()
    outputs, info = prepare(kg_dir)
    assert outputs["review.csv"] == original
    assert info["priority_counts"]["ambiguous_label"] == 1
    assert check(kg_dir, review_dir) == {"accept": 0, "reject": 0, "defer": 0, "pending": 150}
    with pytest.raises(ValueError, match="150 pending"):
        check(kg_dir, review_dir, require_complete=True)

    samples = list(csv.DictReader(io.StringIO(outputs["evidence_samples.csv"].decode())))
    dog = next(row for row in samples if row["class_id"] == "1")
    _update_rows(
        review_dir / "review.csv",
        {
            1: {
                "decision": "accept",
                "selected_uri": "/c/en/dog",
                "reviewer": "Mai",
                "evidence_reference": dog["assertion_uri"],
                "rationale": "The URI describes the object class.",
            },
            2: {
                "decision": "defer",
                "reviewer": "Mai",
                "evidence_reference": "/a/bike",
                "rationale": "Need to inspect ambiguous uses of bike.",
            },
        },
    )
    assert check(kg_dir, review_dir) == {"accept": 1, "reject": 0, "defer": 1, "pending": 148}


def test_rejects_wrong_uri_evidence_and_source_change(tmp_path):
    kg_dir, review_dir = _fixture(tmp_path)
    path = review_dir / "review.csv"
    review = {
        "decision": "accept",
        "selected_uri": "/c/en/cat",
        "reviewer": "Mai",
        "evidence_reference": "/a/dog",
        "rationale": "Inspected the candidate.",
    }
    _update_rows(path, {1: review})
    with pytest.raises(ValueError, match="listed candidate"):
        check(kg_dir, review_dir)
    review["selected_uri"] = "/c/en/dog"
    review["evidence_reference"] = "/a/bike"
    _update_rows(path, {1: review})
    with pytest.raises(ValueError, match="matching assertion URI"):
        check(kg_dir, review_dir)
    review["evidence_reference"] = "/a/dog"
    _update_rows(path, {1: review})
    assert check(kg_dir, review_dir)["accept"] == 1
    metadata = kg_dir / "metadata.json"
    data = json.loads(metadata.read_text())
    data["source_version"] = "tampered"
    metadata.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="source hash mismatch"):
        check(kg_dir, review_dir)


def test_compile_gate_and_class_export(tmp_path):
    kg_dir, review_dir = _fixture(tmp_path)
    rows = {}
    for i in range(1, 151):
        rows[i] = {
            "decision": "accept" if i in (1, 2) else "reject",
            "selected_uri": "/c/en/dog" if i == 1 else "/c/en/bike/n" if i == 2 else "",
            "reviewer": "Mai",
            "evidence_reference": "/a/dog" if i == 1 else "/a/bike" if i == 2 else "no_candidate_in_snapshot",
            "rationale": "Synthetic test decision.",
        }
    _update_rows(review_dir / "review.csv", rows)
    assert check(kg_dir, review_dir, require_complete=True)["accept"] == 2
    reviewed_dir = tmp_path / "reviewed"
    outputs = compile_reviewed(kg_dir, review_dir, reviewed_dir)
    write_outputs(reviewed_dir, outputs)
    kg = load_kg(reviewed_dir)
    review_manifest = reviewed_dir / "review_manifest.json"
    check_gate("reviewed", kg, reviewed_dir, review_manifest)
    assert kg.metadata["resolved_class_count"] == 2
    assert kg.metadata["human_verified_class_count"] == 150

    vocabulary = tmp_path / "vocab.json"
    labels = {1: "dog", 2: "bike"}
    vocabulary.write_text(json.dumps({"label_to_idx": {labels.get(i, f"class {i}"): i for i in range(1, 151)}}))
    ckpt = tmp_path / "best.pt"
    torch.save(
        {
            "manifest": {
                "milestone": "C.1",
                "mode": "reviewed",
                "kg_source": "conceptnet",
                "review_status": "human_approved",
                "review_basis": "individual_conceptnet_class_uri_selection",
                "review_manifest_sha256": sha256(review_manifest),
                "input_sha256": kg.hashes,
                "dim": 8,
            },
            "model": TransE(len(kg.entities), len(kg.relations), 8).state_dict(),
            "epoch": 1,
        },
        ckpt,
    )
    output = tmp_path / "features"
    manifest = export_conceptnet(ckpt, reviewed_dir, vocabulary, output, review_manifest_path=review_manifest)
    features, valid, loaded = load_class_features(output)
    assert manifest == loaded
    assert features.shape == (151, 8)
    assert valid[:3].tolist() == [False, True, True]
    assert not valid[3:].any()

    ids = reviewed_dir / "class_entity_ids.json"
    ids.write_text(ids.read_text() + "\n")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        check_gate("reviewed", load_kg(reviewed_dir), reviewed_dir, review_manifest)
    ids.write_text(ids.read_text().rstrip("\n") + "\n")
    candidates = reviewed_dir / "class_concept_candidates.json"
    candidates.write_text(candidates.read_text() + "\n")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        check_gate("reviewed", load_kg(reviewed_dir), reviewed_dir, review_manifest)
