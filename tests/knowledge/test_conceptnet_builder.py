"""B2 ConceptNet smoke tests are strictly synthetic: no download or VG annotations."""

import copy
import csv
import gzip
import hashlib
import io
import json

import pytest

from scripts.build_conceptnet_graph import write_outputs
from src.knowledge.conceptnet_builder import _concept_key, extract_assertions


def mapping():
    rows = [{"class_id": 0, "label": "__background__", "status": "background", "synset_id": None}]
    for i in range(1, 151):
        label = {1: "dog", 2: "bike", 3: "banana", 4: "unknown"}.get(i, f"class {i}")
        rows.append(
            {
                "class_id": i,
                "label": label,
                "status": "ambiguous" if i != 1 else "linked_auto",
                "synset_id": "dog.n.01" if i == 1 else None,
            }
        )
    return {"schema_version": 1, "source": "wordnet", "source_version": "3.0", "rows": rows}


def assertion(relation, start, end, *, dataset="/d/omcs", license_id="cc:by-sa/4.0", weight=2.0):
    uri = f"/a/[{relation},{start},{end}]"
    metadata = {"dataset": dataset, "license": license_id, "weight": weight}
    return "\t".join((uri, relation, start, end, json.dumps(metadata, sort_keys=True))) + "\n"


def sample():
    # Synthetic assertions are NOT quotations or evidence from ConceptNet.
    return "".join(
        [
            assertion("/r/CapableOf", "/c/en/dog", "/c/en/bark"),
            assertion("/r/AtLocation", "/c/en/park", "/c/en/dog/n"),
            assertion("/r/IsA", "/c/en/bike/n", "/c/en/vehicle"),
            assertion("/r/UsedFor", "/c/en/banana", "/c/en/eat"),
            assertion("/r/Antonym", "/c/en/dog", "/c/en/cat"),
            assertion("/r/CapableOf", "/c/en/horse", "/c/en/run"),
            assertion("/r/IsA", "/c/ja/dog", "/c/en/animal"),
            assertion("/r/IsA", "/c/en/dog", "/c/en/animal", dataset="/d/wordnet/3.1"),
            assertion("/r/HasA", "/c/en/dog", "/c/en/teeth", license_id="non-standard"),
            assertion("/r/RelatedTo", "/c/en/dog", "/c/en/pet", weight=0.2),
            assertion("/r/CapableOf", "/c/en/dog/v", "/c/en/follow"),
        ]
    )


def make_file(tmp_path, *, compressed=False):
    path = tmp_path / ("conceptnet-5.7.0.tsv.gz" if compressed else "conceptnet-5.7.0.tsv")
    payload = sample().encode()
    data = gzip.compress(payload, mtime=0) if compressed else payload
    path.write_bytes(data)
    return path, hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize("compressed", [False, True])
def test_extract_reproducible_and_no_false_alignment(tmp_path, compressed):
    path, digest = make_file(tmp_path, compressed=compressed)
    files, meta = extract_assertions(mapping(), path, expected_sha256=digest)
    files2, meta2 = extract_assertions(mapping(), path, expected_sha256=digest)
    assert files == files2 and meta == meta2
    assert meta["source_version"] == "5.7.0"
    assert meta["class_count_including_background"] == 151
    assert meta["class_count_with_lexical_candidates"] == 3
    assert meta["edge_evidence_count"] == 4
    assert meta["unique_triple_count"] == 4
    assert meta["filter_counts"]["input_rows"] == 11
    assert meta["filter_counts"]["filtered_wordnet_derived"] == 1
    assert meta["wordnet_conceptnet_sameas_links"] == 0
    assert meta["human_verified_class_count"] == 0
    assert meta["ready_for_gat_kg_training"] is False
    assert "cn:/c/en/dog" in json.loads(files["entities.json"])["entities"]
    assert not any(node.startswith("wn:") for node in json.loads(files["entities.json"])["entities"])
    classes = json.loads(files["class_concept_candidates.json"])["classes"]
    assert len(classes) == 151
    assert classes[0]["concept_ids"] == []
    assert classes[1]["concept_uris"] == ["/c/en/dog", "/c/en/dog/n"]
    assert classes[1]["human_verified"] is False
    assert classes[4]["concept_ids"] == []
    review = list(csv.DictReader(io.StringIO(files["review_queue.csv"].decode())))
    assert len(review) == 150
    assert all(not r["reviewer"] and not r["sense_alignment_decision"] for r in review)
    triples = list(csv.DictReader(io.StringIO(files["triples.tsv"].decode()), delimiter="\t"))
    assert len(triples) == 4
    assert all(row["dataset"] == "/d/omcs" for row in triples)
    assert all(row["license"] == "cc:by-sa/4.0" for row in triples)


def test_sha256_and_limits_are_enforced(tmp_path):
    path, digest = make_file(tmp_path)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        extract_assertions(mapping(), path, expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="64 hexadecimal"):
        extract_assertions(mapping(), path, expected_sha256="abc")
    with pytest.raises(ValueError, match="edge limit"):
        extract_assertions(mapping(), path, expected_sha256=digest, max_edges=1)
    with pytest.raises(ValueError, match="Invalid weight"):
        extract_assertions(mapping(), path, expected_sha256=digest, min_weight=float("nan"))
    with pytest.raises(ValueError, match="No ConceptNet assertions"):
        extract_assertions(mapping(), path, expected_sha256=digest, min_weight=2.5)


def test_missing_and_invalid_vocabulary_rows(tmp_path):
    path, digest = make_file(tmp_path)
    bad = mapping()
    bad["rows"][1]["class_id"] = 3
    with pytest.raises(ValueError, match="ordering"):
        extract_assertions(bad, path, expected_sha256=digest)
    bad = mapping()
    bad["rows"] = bad["rows"][:-1]
    with pytest.raises(ValueError, match="151"):
        extract_assertions(bad, path, expected_sha256=digest)
    bad = copy.deepcopy(mapping())
    bad["rows"][2]["label"] = "dog"
    with pytest.raises(ValueError, match="duplicate"):
        extract_assertions(bad, path, expected_sha256=digest)


def test_complex_uris_do_not_fake_lexical_identity():
    assert _concept_key("/c/en/dog/n") == "dog"
    assert _concept_key("/c/en/dog/n/wn/animal") is None
    assert _concept_key("/c/en/dog%2Fcat") is None
    assert _concept_key("/c/ja/dog") is None
    assert _concept_key("/c/en/dog/v", allow_pos=False) is None


def test_refuses_to_overwrite_outputs_and_symlinks(tmp_path):
    out = tmp_path / "generated"
    write_outputs(out, {"data.json": b"good"}, dry_run=True)
    assert not out.exists()
    write_outputs(out, {"data.json": b"good"})
    assert (out / "data.json").read_bytes() == b"good"
    write_outputs(out, {"data.json": b"good"})
    with pytest.raises(FileExistsError, match="different existing"):
        write_outputs(out, {"data.json": b"bad", "new.json": b"none"})
    assert not (out / "new.json").exists()
    link = tmp_path / "alias"
    link.symlink_to(out, target_is_directory=True)
    with pytest.raises(FileExistsError, match="symlink"):
        write_outputs(link, {"new.json": b"bad"})
