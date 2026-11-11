"""C.2 contract tests using invented vocabulary, mapping and graph only."""

import json

import pytest
import torch

from src.knowledge.export import export_wordnet, load_class_features
from src.knowledge.transe.core import TransE, load_kg, sha256


@pytest.fixture
def bundle(tmp_path):
    kg = tmp_path / "kg"
    kg.mkdir()
    vocabulary = tmp_path / "vocab.json"
    vocabulary.write_text(json.dumps({"label_to_idx": {f"class{i}": i for i in range(1, 151)}}))
    mapping = tmp_path / "mapping.json"
    rows = [{"class_id": 0, "label": "__background__", "status": "background", "synset_id": None}]
    for i in range(1, 151):
        rows.append(
            {
                "class_id": i,
                "label": f"class{i}",
                "status": "linked_manual" if i < 150 else "ambiguous",
                "synset_id": f"concept{i % 12}.n.01" if i < 150 else None,
                "candidates": [],
                "curation_tier": "AI_PROVISIONAL_NOT_HUMAN_VERIFIED",
            }
        )
    mapping.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source": "wordnet",
                "source_version": "synthetic",
                "vocabulary_sha256": sha256(vocabulary),
                "rows": rows,
            }
        )
    )
    (kg / "entities.json").write_text(json.dumps({"entities": [f"wn:concept{i}.n.01" for i in range(12)]}))
    (kg / "relations.json").write_text(json.dumps({"relations": ["IsA"]}))
    (kg / "triples.tsv").write_text("subject_id\trelation_id\tobject_id\n0\t0\t1\n")
    (kg / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source": "wordnet",
                "source_version": "synthetic",
                "triple_count": 1,
                "input_mapping_sha256": sha256(mapping),
            }
        )
    )
    (kg / "class_entity_ids.json").write_text(
        json.dumps({"class_entity_ids": [None] + [i % 12 for i in range(1, 150)] + [None]})
    )
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "manifest": {
                "milestone": "C.1",
                "mode": "draft",
                "kg_source": "wordnet",
                "input_sha256": load_kg(kg).hashes,
                "dim": 8,
            },
            "model": TransE(12, 1, 8).state_dict(),
            "epoch": 1,
        },
        checkpoint,
    )
    return checkpoint, kg, mapping, vocabulary, tmp_path / "export"


def test_export_alignment_mask_and_determinism(bundle):
    ck, kg, mapping, vocab, output = bundle
    manifest = export_wordnet(*bundle)
    features, valid, _ = load_class_features(output, allow_draft=True)
    weights = torch.load(ck, weights_only=True)["model"]["entity.weight"]
    assert features.shape == (151, 8)
    assert valid.sum() == 149 and not valid[0] and not valid[150]
    assert torch.equal(features[1], weights[1])
    assert torch.equal(features[12], weights[0])
    assert torch.equal(features[1], features[13])  # Shared synsets are allowed.
    assert torch.count_nonzero(features[[0, 150]]) == 0
    assert all(r["review_status"] != "human_approved" for r in manifest["rows"])
    export_wordnet(ck, kg, mapping, vocab, output.with_name("again"))
    other, other_mask, _ = load_class_features(output.with_name("again"), allow_draft=True)
    assert torch.equal(features, other) and torch.equal(valid, other_mask)
    with pytest.raises(FileExistsError):
        export_wordnet(*bundle)
    with pytest.raises(ValueError, match="allow-draft"):
        load_class_features(output)


@pytest.mark.parametrize("mutation", ["kg", "mapping", "vocab", "class_ids", "shape", "nan", "reviewed"])
def test_export_rejects_mismatches(bundle, mutation):
    ck, kg, mapping, vocab, output = bundle
    if mutation == "kg":
        p = kg / "entities.json"
        d = json.loads(p.read_text())
        d["entities"].reverse()
        p.write_text(json.dumps(d))
    elif mutation in ("mapping", "vocab"):
        p = mapping if mutation == "mapping" else vocab
        p.write_text(p.read_text() + "\n")
    elif mutation == "class_ids":
        p = kg / "class_entity_ids.json"
        d = json.loads(p.read_text())
        d["class_entity_ids"][1] = 0
        p.write_text(json.dumps(d))
    else:
        d = torch.load(ck, weights_only=True)
        if mutation == "shape":
            d["model"]["entity.weight"] = torch.zeros(11, 8)
        elif mutation == "nan":
            d["model"]["entity.weight"][0, 0] = float("nan")
        else:
            d["manifest"]["mode"] = "reviewed"
        torch.save(d, ck)
    with pytest.raises(ValueError):
        export_wordnet(*bundle)
    assert not output.exists()


def test_loader_rejects_tampering(bundle):
    export_wordnet(*bundle)
    output = bundle[-1]
    p = output / "class_embeddings.pt"
    p.write_bytes(p.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="SHA256"):
        load_class_features(output, allow_draft=True)


def test_loader_rejects_row_reordering(bundle):
    export_wordnet(*bundle)
    p = bundle[-1] / "manifest.json"
    d = json.loads(p.read_text())
    d["rows"][1], d["rows"][2] = d["rows"][2], d["rows"][1]
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="row order"):
        load_class_features(bundle[-1], allow_draft=True)


def test_reviewed_export_loads_without_allow_draft(bundle):
    ck, kg, mapping, vocab, output = bundle
    mapping_doc = json.loads(mapping.read_text())
    mapping_doc["rows"][150]["status"] = "linked_manual"
    mapping_doc["rows"][150]["synset_id"] = "concept6.n.01"
    mapping.write_text(json.dumps(mapping_doc))
    ids = json.loads((kg / "class_entity_ids.json").read_text())
    ids["class_entity_ids"][150] = 6
    (kg / "class_entity_ids.json").write_text(json.dumps(ids))
    metadata = json.loads((kg / "metadata.json").read_text())
    metadata.update(ready_for_gat_kg_training=True, resolved_class_count=150, input_mapping_sha256=sha256(mapping))
    (kg / "metadata.json").write_text(json.dumps(metadata))
    review = output.with_name("human_review.json")
    review.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "review_status": "human_approved",
                "kg_source": "wordnet",
                "kg_path": str(kg.resolve()),
                "kg_sha256": load_kg(kg).hashes,
                "reviews": [
                    {
                        "class_id": i,
                        "decision": "accept",
                        "reviewer": "Toan",
                        "evidence_reference": f"selection:{i}",
                        "rationale": "Synthetic test rationale",
                    }
                    for i in range(1, 151)
                ],
            }
        )
    )
    checkpoint = torch.load(ck, weights_only=True)
    checkpoint["manifest"].update(
        mode="reviewed",
        review_status="human_approved",
        review_manifest_sha256=sha256(review),
        input_sha256=load_kg(kg).hashes,
    )
    torch.save(checkpoint, ck)

    manifest = export_wordnet(ck, kg, mapping, vocab, output, review_manifest_path=review)
    features, valid, loaded = load_class_features(output)
    assert manifest["mode"] == loaded["mode"] == "reviewed"
    assert manifest["review_status"] == "human_approved"
    assert features.shape == (151, 8) and int(valid.sum()) == 150
    (output / "review_manifest.json").write_text("tampered")
    with pytest.raises(ValueError, match="manifest SHA256 mismatch"):
        load_class_features(output)
