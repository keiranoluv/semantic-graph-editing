"""B.2.2 candidate alignment tests with synthetic 151-class KG artifacts."""

import csv
import hashlib
import io
import json

import pytest

from src.knowledge.alignment_candidates import _canonical_sha, build_candidates, load_and_build


def to_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def fixtures():
    rows = [
        {
            "class_id": 0,
            "label": "__background__",
            "status": "background",
            "synset_id": None,
        }
    ]
    wn_entities = {"entities": []}
    class_ids = [None]
    cn_entities = {"entities": [f"cn:/c/en/class_{i}" for i in range(1, 151)] + ["cn:/c/en/class_1/n"]}
    cn_classes = [
        {"class_id": 0, "label": "__background__", "concept_uris": [], "concept_ids": [], "human_verified": False}
    ]
    for i in range(1, 151):
        linked = i <= 124
        synset = f"class_{i}.n.01" if linked else None
        rows.append(
            {
                "class_id": i,
                "label": f"class {i}",
                "status": "linked_manual" if linked else "ambiguous",
                "synset_id": synset,
                "curation_tier": "AI_PROVISIONAL_NOT_HUMAN_VERIFIED" if linked else None,
            }
        )
        class_ids.append(i - 1 if linked else None)
        if linked:
            wn_entities["entities"].append(f"wn:{synset}")
        uris = [f"/c/en/class_{i}"] + (["/c/en/class_1/n"] if i == 1 else [])
        ids = [i - 1] + ([150] if i == 1 else [])
        cn_classes.append(
            {
                "class_id": i,
                "label": f"class {i}",
                "concept_uris": uris,
                "concept_ids": ids,
                "human_verified": False,
            }
        )
    mapping = {
        "schema_version": 1,
        "source": "wordnet",
        "source_version": "3.0",
        "curation_tier": "AI_PROVISIONAL_NOT_HUMAN_VERIFIED",
        "rows": rows,
    }
    wn_meta = {
        "schema_version": 1,
        "source": "wordnet",
        "source_version": "3.0",
        "input_mapping_sha256": hashlib.sha256(to_bytes(mapping)).hexdigest(),
        "resolved_class_count": 124,
        "ready_for_gat_kg_training": False,
    }
    cn_meta = {
        "schema_version": 1,
        "source": "conceptnet",
        "source_version": "5.7.0",
        "input_mapping_canonical_sha256": _canonical_sha(mapping),
        "class_count_with_lexical_candidates": 150,
        "human_verified_class_count": 0,
        "ready_for_gat_kg_training": False,
        "source_snapshot_sha256": "f" * 64,
    }
    return (
        mapping,
        wn_entities,
        {"class_entity_ids": class_ids},
        wn_meta,
        cn_entities,
        {"classes": cn_classes},
        cn_meta,
    )


def run_fixture(*args):
    values = args or fixtures()
    return build_candidates(
        *values,
        mapping_bytes_sha256=hashlib.sha256(to_bytes(values[0])).hexdigest(),
        input_sha256={"wordnet_mapping": "x" * 64},
    )


def test_candidates_are_always_unverified_and_have_151_class_rows():
    files, metadata = run_fixture()
    assert metadata["wordnet_linked_class_count"] == 124
    assert metadata["conceptnet_candidate_class_count"] == 150
    assert metadata["both_candidate_class_count"] == 124
    assert metadata["candidate_pair_count"] == 125
    assert metadata["wordnet_missing_class_ids"] == list(range(125, 151))
    assert metadata["human_verified_bridge_count"] == metadata["sameas_edge_count"] == 0
    assert metadata["ready_for_gat_kg_training"] is False
    classes = json.loads(files["class_alignment_candidates.json"])["classes"]
    assert len(classes) == 151
    assert classes[0]["conceptnet_candidates"] == []
    assert classes[125]["wordnet_synset"] is None
    assert len(classes[125]["conceptnet_candidates"]) == 1
    pairs = list(csv.DictReader(io.StringIO(files["candidate_pairs.tsv"].decode()), delimiter="\t"))
    assert len(pairs) == 125
    assert all(row["candidate_basis"] == "SHARED_VG150_LABEL_ONLY_NOT_IDENTITY" for row in pairs)
    review = list(csv.DictReader(io.StringIO(files["review_queue.csv"].decode())))
    assert len(review) == 150
    assert all(not row["human_reviewer"] and not row["bridge_review_decision"] for row in review)
    pair_review = list(csv.DictReader(io.StringIO(files["bridge_review_queue.csv"].decode())))
    assert len(pair_review) == 125
    assert all(not row["bridge_decision"] and not row["human_reviewer"] for row in pair_review)
    assert files == run_fixture()[0]  # Determinism on fixed inputs.


def test_rejects_mismatched_kg_mapping_hashes():
    a = fixtures()
    a[3]["input_mapping_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="WordNet KG input mapping SHA256 mismatch"):
        run_fixture(*a)
    a = fixtures()
    a[6]["input_mapping_canonical_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="ConceptNet input mapping SHA256 mismatch"):
        run_fixture(*a)


def test_rejects_wrong_class_and_node_ids():
    a = fixtures()
    a[5]["classes"][2]["label"] = "not class 2"
    with pytest.raises(ValueError, match="Class label mismatch"):
        run_fixture(*a)
    a = fixtures()
    a[2]["class_entity_ids"][10] = 100
    with pytest.raises(ValueError, match="WordNet class alignment mismatch"):
        run_fixture(*a)
    a = fixtures()
    a[5]["classes"][10]["concept_ids"] = [11]
    with pytest.raises(ValueError, match="ConceptNet class alignment mismatch"):
        run_fixture(*a)


def test_never_claims_human_review():
    a = fixtures()
    a[5]["classes"][1]["human_verified"] = True
    with pytest.raises(ValueError, match="human-verified"):
        run_fixture(*a)
    a = fixtures()
    a[0]["curation_tier"] = "HUMAN_REVIEWED"
    a[3]["input_mapping_sha256"] = hashlib.sha256(to_bytes(a[0])).hexdigest()
    a[6]["input_mapping_canonical_sha256"] = _canonical_sha(a[0])
    with pytest.raises(ValueError, match="AI-provisional"):
        run_fixture(*a)
    a = fixtures()
    a[6]["ready_for_gat_kg_training"] = True
    with pytest.raises(ValueError, match="non-training-ready"):
        run_fixture(*a)


def test_rejects_missing_or_bad_background():
    a = fixtures()
    a[2]["class_entity_ids"][0] = 0
    with pytest.raises(ValueError, match="Background WordNet ID"):
        run_fixture(*a)
    a = fixtures()
    a[5]["classes"][0]["concept_uris"] = ["/c/en/background"]
    a[5]["classes"][0]["concept_ids"] = [0]
    with pytest.raises(ValueError, match="Background ConceptNet candidates"):
        run_fixture(*a)


def test_disk_loader_rejects_tampered_mapping(tmp_path):
    a = fixtures()
    mapping_file = tmp_path / "mapping.json"
    wn_dir, cn_dir = tmp_path / "wn", tmp_path / "cn"
    wn_dir.mkdir()
    cn_dir.mkdir()
    mapping_file.write_bytes(to_bytes(a[0]))
    for path, obj in [
        (wn_dir / "entities.json", a[1]),
        (wn_dir / "class_entity_ids.json", a[2]),
        (wn_dir / "metadata.json", a[3]),
        (cn_dir / "entities.json", a[4]),
        (cn_dir / "class_concept_candidates.json", a[5]),
        (cn_dir / "metadata.json", a[6]),
    ]:
        path.write_bytes(to_bytes(obj))
    outputs, info = load_and_build(wordnet_mapping=mapping_file, wordnet_kg=wn_dir, conceptnet_kg=cn_dir)
    assert info["candidate_pair_count"] == 125
    assert len(info["input_file_sha256"]) == 7
    a[0]["rows"][1]["label"] = "tampered"
    mapping_file.write_bytes(to_bytes(a[0]))
    with pytest.raises(ValueError, match="WordNet KG input mapping SHA256 mismatch"):
        load_and_build(wordnet_mapping=mapping_file, wordnet_kg=wn_dir, conceptnet_kg=cn_dir)
