"""GQA IDs, missing embeddings and frozen-reference safety must fail closed."""

import copy
import json

import pytest
import torch

from src.knowledge.transe.core import TransE, load_kg, sha256
from tools.export_gqa_wordnet import export_features, load_features
from tools.prepare_gqa_wordnet import review_packet, transfer_overrides


def test_transfer_uses_exact_labels_not_original_ids_or_inferred_plural():
    vocabulary = {"label_to_idx": {"dog": 3, "dogs": 1, "cat": 2}}
    mapping = {"rows": [{}, {"class_id": 1, "label": "dog", "status": "linked_manual", "synset_id": "dog.n.01"}]}
    overrides = {"mappings": {"dog": "dog.n.01"}, "review_provenance": {"dog": {"reviewer": "Human"}}}
    result = transfer_overrides(vocabulary, mapping, overrides)
    assert result["mappings"] == {"dog": "dog.n.01"}
    assert result["transfer_provenance"]["dog"]["gqa_class_id"] == 3
    assert result["transfer_provenance"]["dog"]["source_class_id"] == 1
    overrides["review_provenance"] = {}
    with pytest.raises(ValueError, match="review evidence"):
        transfer_overrides(vocabulary, mapping, overrides)


@pytest.fixture
def gqa_bundle(tmp_path):
    folder = tmp_path / "mapping"
    kg = folder / "kg"
    kg.mkdir(parents=True)
    manifest_path = tmp_path / "gqa.json"
    labels = {"trained": 1, "isolated": 2, "unknown": 3}
    manifest_path.write_text(json.dumps({"vocabulary": {"label_to_idx": labels}}))
    (folder / "vocabulary.json").write_text(json.dumps({"label_to_idx": labels}))
    rows = [{"class_id": 0, "label": "__background__", "status": "background", "synset_id": None, "candidates": []}]
    for i, label in enumerate(labels, 1):
        synset = f"{label}.n.01" if i < 3 else None
        rows.append(
            {
                "class_id": i,
                "label": label,
                "status": "linked_auto" if i < 3 else "unmatched",
                "synset_id": synset,
                "candidates": [{"synset_id": synset, "definition": "a test noun"}] if synset else [],
            }
        )
    mapping = {
        "schema_version": 1,
        "source": "wordnet",
        "source_version": "synthetic",
        "rows": rows,
        "vocabulary_sha256": sha256(folder / "vocabulary.json"),
    }
    (folder / "wordnet_mapping.json").write_text(json.dumps(mapping))
    (folder / "coverage.json").write_text(json.dumps({"sources": {"gqa_manifest_sha256": sha256(manifest_path)}}))
    entities = ["wn:trained.n.01", "wn:isolated.n.01"] + [f"wn:node{i}.n.01" for i in range(6)]
    (kg / "entities.json").write_text(json.dumps({"entities": entities}))
    (kg / "relations.json").write_text(json.dumps({"relations": ["IsA"]}))
    triples = [(i, 0, j) for i in range(8) for j in range(8) if i != j and i != 1 and j != 1]
    (kg / "triples.tsv").write_text(
        "subject_id\trelation_id\tobject_id\n" + "".join(f"{h}\t{r}\t{t}\n" for h, r, t in triples)
    )
    (kg / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source": "wordnet",
                "triple_count": len(triples),
                "input_mapping_sha256": sha256(folder / "wordnet_mapping.json"),
            }
        )
    )
    (kg / "class_entity_ids.json").write_text(json.dumps({"class_entity_ids": [None, 0, 1, None]}))
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "manifest": {
                "milestone": "C.1",
                "mode": "draft",
                "kg_source": "wordnet",
                "split_policy": "seeded_transductive_coverage_holdout",
                "seed": 42,
                "val_ratio": 0.1,
                "test_ratio": 0.1,
                "dim": 8,
                "input_sha256": load_kg(kg).hashes,
            },
            "model": TransE(8, 1, 8).state_dict(),
            "epoch": 1,
        },
        checkpoint,
    )
    return checkpoint, folder, tmp_path / "export", manifest_path, mapping


def test_export_masks_isolated_and_unresolved_entities_and_preserves_alignment(gqa_bundle):
    checkpoint, folder, output, manifest_path, _ = gqa_bundle
    report = export_features(checkpoint, folder, output)
    features, valid, _ = load_features(output, manifest_path)
    assert features.shape == (4, 8)
    assert valid.tolist() == [False, True, False, False]
    weights = torch.load(checkpoint, weights_only=True)["model"]["entity.weight"]
    assert torch.equal(features[1], weights[0])
    assert torch.count_nonzero(features[~valid]) == 0
    assert report["rows"][2]["reason"] == "isolated_in_kg_training"
    assert report["gqa_context_human_review_completed"] is False
    with pytest.raises(FileExistsError):
        export_features(checkpoint, folder, output)


@pytest.mark.parametrize("mutation", ["alignment", "mapping", "vocabulary", "kg", "weights"])
def test_export_rejects_wrong_provenance(gqa_bundle, mutation):
    checkpoint, folder, output, _, _ = gqa_bundle
    if mutation == "alignment":
        (folder / "kg/class_entity_ids.json").write_text(json.dumps({"class_entity_ids": [None, 1, 0, None]}))
    elif mutation == "weights":
        payload = torch.load(checkpoint, weights_only=True)
        payload["model"]["entity.weight"][0, 0] = float("nan")
        torch.save(payload, checkpoint)
    else:
        path = (
            folder
            / {"mapping": "wordnet_mapping.json", "vocabulary": "vocabulary.json", "kg": "kg/entities.json"}[mutation]
        )
        path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError):
        export_features(checkpoint, folder, output)
    assert not output.exists()


@pytest.mark.parametrize("mutation", ["tensor", "vocab", "rows", "missing_nonzero"])
def test_loader_rejects_changed_features(gqa_bundle, mutation):
    checkpoint, folder, output, manifest_path, _ = gqa_bundle
    export_features(checkpoint, folder, output)
    metadata_path = output / "manifest.json"
    metadata = json.loads(metadata_path.read_text())
    if mutation == "tensor":
        with (output / "class_embeddings.pt").open("ab") as f:
            f.write(b"changed")
    elif mutation == "vocab":
        manifest_path.write_text(manifest_path.read_text() + "\n")
    elif mutation == "rows":
        metadata["rows"][1], metadata["rows"][2] = metadata["rows"][2], metadata["rows"][1]
    else:
        path = output / "class_embeddings.pt"
        data = torch.load(path, weights_only=True)
        data["class_embeddings"][2] = 1
        torch.save(data, path)
        metadata["tensor_sha256"] = sha256(path)
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        load_features(output, manifest_path)


def test_review_sample_is_deterministic_without_fabricating_human_decisions(gqa_bundle):
    mapping = gqa_bundle[-1]
    original = copy.deepcopy(mapping)
    all_rows, sample = review_packet(mapping, sample_size=2)
    assert review_packet(mapping, sample_size=2) == (all_rows, sample)
    assert mapping == original and len({r["class_id"] for r in sample}) == 2
    assert all(r["reviewer"] == r["decision"] == r["selected_synset"] == "" for r in all_rows)
