"""Review evidence uses train records only; submissions require explicit choices."""

import csv
import io
import json
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import h5py
import pytest
from PIL import Image

from src.knowledge.transe.core import sha256
from tests.knowledge.test_entity_linking import FakeWordNet
from tools import prepare_gqa_wordnet_review as review


def record(image_id, width=200, height=160):
    return {
        "image_id": image_id,
        "width": width,
        "height": height,
        "objects": [
            {"object_id": 9, "label_id": 1, "bbox_xyxy": [20, 40, 100, 120]},
            {"object_id": 8, "label_id": 1, "bbox_xyxy": [20, 40, 100, 120]},
            {"object_id": 1, "label_id": 1, "bbox_xyxy": [0, 0, 2, 2]},
        ],
    }


def test_seeded_selection_distinct_images_largest_instance_and_order_independent():
    records = [record(str(i)) for i in range(20)]
    classes = [{"class_id": 1}, {"class_id": 2}]
    chosen, counts, invalid = review.select_examples(records, classes, per_class=5)
    assert review.select_examples(list(reversed(records)), classes, per_class=5) == (chosen, counts, invalid)
    assert len(chosen[1]) == len({e["image_id"] for e in chosen[1]}) == 5
    assert all(e["object_id"] == 8 for e in chosen[1])
    assert counts[1] == 20 and chosen[2] == []
    assert review.select_examples(records, classes, per_class=5, seed=7)[0][1] != chosen[1]
    with pytest.raises(ValueError, match="3..5"):
        review.select_examples(records, classes, per_class=2)


def test_invalid_boxes_are_reported_and_missing_evidence_not_fabricated():
    bad = record("bad")
    for obj in bad["objects"]:
        obj["bbox_xyxy"] = [1, 1, 1, 1]
    chosen, counts, invalid = review.select_examples([bad], [{"class_id": 1}])
    assert chosen[1] == [] and counts[1] == 0 and invalid[1] == 3


def test_real_pixels_are_preserved_and_box_scaled_to_jpeg():
    stream = io.BytesIO()
    Image.new("RGB", (100, 80), (10, 130, 20)).save(stream, format="JPEG")
    original = stream.getvalue()
    example = {"annotation_size": [200, 160], "annotation_box_xyxy": [20, 40, 100, 120]}
    (context, crop), metadata = review.render_evidence(original, example)
    assert metadata["jpeg_box_xyxy"] == [10, 20, 50, 60]
    assert metadata["crop_bounds_xyxy"] == [10, 20, 50, 60]
    with Image.open(io.BytesIO(crop)) as image:
        assert image.size == (40, 40)
        assert image.getpixel((20, 20))[1] > 100
    with Image.open(io.BytesIO(context)) as image:
        red, green, _ = image.getpixel((10, 30))
        assert red > green + 60
        assert image.getpixel((30, 40))[1] > 100
    assert original == stream.getvalue()


@pytest.fixture
def train_cache(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "splits": {"train": ["t1", "t2"], "val": ["v1"], "test": ["heldout"]},
        "annotation_sha256": {"train_sceneGraphs.json": "train-source"},
    }
    manifest_path.write_text(json.dumps(manifest))
    identity = {
        "manifest_sha256": sha256(manifest_path),
        "splits": manifest["splits"],
        "annotation_sha256": "train-source",
        "image_archive_sha256": "archive-source",
    }
    identity_hash = review.canonical_hash(identity)
    shard = tmp_path / "train.h5"
    with h5py.File(shard, "w") as h5:
        h5.attrs["identity_sha256"] = identity_hash
        for image_id in manifest["splits"]["train"]:
            h5.create_group(image_id).create_dataset("record", data=json.dumps(record(image_id)).encode())
    entries = [
        {"file": "train.h5", "ids": ["t1", "t2"], "sha256": sha256(shard), "identity_sha256": identity_hash},
        {"file": "validation-MUST-NOT-OPEN.h5", "ids": ["v1"], "sha256": "forbidden", "identity_sha256": identity_hash},
    ]
    (tmp_path / "manifest.json").rename(tmp_path / "gqa.json")
    manifest_path = tmp_path / "gqa.json"
    (tmp_path / "manifest.json").write_text(json.dumps({"identity": identity, "complete": True, "shards": entries}))
    return tmp_path, manifest, manifest_path


def test_train_shards_only_no_validation_annotation_or_image_records_opened(train_cache, monkeypatch):
    folder, manifest, path = train_cache
    original = review.sha256

    def guarded(file):
        assert "validation" not in str(file) and "sceneGraphs" not in str(file)
        return original(file)

    monkeypatch.setattr(review, "sha256", guarded)
    provenance = {}
    records = list(review.train_records(folder, manifest, path, provenance))
    assert [r["image_id"] for r in records] == ["t1", "t2"]
    assert provenance["train_images_scanned"] == 2
    assert len(provenance["train_shards_verified"]) == 1


def test_mixed_train_validation_shard_is_rejected_before_opening(train_cache):
    folder, manifest, path = train_cache
    metadata_path = folder / "manifest.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["shards"][0]["ids"].append("v1")
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="non-train"):
        list(review.train_records(folder, manifest, path, {}))


def test_sample_identity_and_candidate_definitions_are_bound(tmp_path):
    original = {k: "" for k in review.ORIGINAL_FIELDS}
    original.update(
        class_id="1",
        label="dog",
        status="linked_auto",
        synset_id="dog.n.01",
        definition="a noun",
        candidates="dog.n.01: a noun",
    )
    sample, mapping, manifest = [tmp_path / f"{name}.json" for name in ("sample", "mapping", "manifest")]
    sample.write_bytes(_csv([original], review.ORIGINAL_FIELDS))
    mapping.write_text(
        json.dumps(
            {
                "source_version": "test-wordnet",
                "rows": [
                    {},
                    {
                        "class_id": 1,
                        "label": "dog",
                        "status": "linked_auto",
                        "synset_id": "dog.n.01",
                        "candidates": [{"synset_id": "dog.n.01", "definition": "a noun"}],
                    },
                ],
            }
        )
    )
    manifest.write_text(json.dumps({"vocabulary": {"label_to_idx": {"dog": 1}}}))
    _, classes = review.load_sample(sample, mapping, manifest, FakeWordNet())
    assert classes[0]["original"] == original
    original["decision"] = "accept"
    sample.write_bytes(_csv([original], review.ORIGINAL_FIELDS))
    with pytest.raises(ValueError, match="blank human"):
        review.load_sample(sample, mapping, manifest, FakeWordNet())


def _csv(rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


@pytest.fixture
def packet():
    classes = []
    for class_id, label in ((20, "fixture animal"), (604, "fixture unsupported")):
        original = {k: "" for k in review.ORIGINAL_FIELDS}
        supported = class_id == 20
        original.update(
            class_id=str(class_id),
            label=label,
            status="ambiguous" if supported else "unmatched",
            candidates="fixture.n.01: synthetic definition" if supported else "",
        )
        classes.append(
            {
                "class_id": class_id,
                "label": label,
                "status": original["status"],
                "synset_id": None,
                "definition": "",
                "candidates": [{"synset_id": "fixture.n.01", "definition": "synthetic definition"}]
                if supported
                else [],
                "examples": [
                    {
                        "image_id": "synthetic-train",
                        "object_id": 1,
                        "annotation_box_xyxy": [0, 0, 10, 10],
                        "annotation_size": [10, 10],
                        "jpeg_size": [10, 10],
                        "context_path": "data/assets/context.jpg",
                        "crop_path": "data/assets/crop.jpg",
                    }
                ],
                "evidence_status": "insufficient_examples",
                "eligible_train_images": 1,
                "image_errors": [],
                "original": original,
            }
        )
    return {"classes": classes, "packet_sha256": "synthetic-fixture-only", "export_fields": list(review.EXPORT_FIELDS)}


def test_empty_export_preserves_original_ids_labels_and_never_invents_approvals(packet):
    content = review.export_csv(packet, {})
    rows = list(csv.DictReader(io.StringIO(content.decode("utf-8-sig"))))
    assert [(r["class_id"], r["label"]) for r in rows] == [("20", "fixture animal"), ("604", "fixture unsupported")]
    assert all(r["decision"] == r["selected_synset"] == r["reviewer"] == "" for r in rows)


def test_explicit_review_roundtrips_unicode_quoted_multiline_notes(packet):
    notes = 'Ảnh 1: "con vật", chưa rõ\nDòng thứ hai'
    content = review.export_csv(
        packet,
        {"20": {"decision": "accept", "selected_synset": "fixture.n.01", "reviewer": "Fixture QA", "notes": notes}},
    )
    rows = list(csv.DictReader(io.StringIO(content.decode("utf-8-sig"))))
    assert rows[0]["notes"] == notes and rows[0]["selected_synset"] == "fixture.n.01"
    assert rows[1]["decision"] == ""


@pytest.mark.parametrize(
    "item",
    [
        {"decision": "accept", "reviewer": "Name"},
        {"decision": "accept", "selected_synset": "fixture.n.01"},
        {"decision": "defer"},
        {"decision": "approved", "reviewer": "Name"},
        {"selected_synset": "not_in_candidates.n.01"},
    ],
)
def test_invalid_decisions_not_exportable(packet, item):
    with pytest.raises(ValueError):
        review.export_csv(packet, {"20": item})


def test_accept_without_images_or_candidate_is_rejected_and_defer_allowed(packet):
    choice = {"decision": "accept", "selected_synset": "fixture.n.01", "reviewer": "Fixture QA"}
    with pytest.raises(ValueError):
        review.export_csv(packet, {"604": choice})
    packet["classes"][0]["examples"] = []
    with pytest.raises(ValueError, match="image evidence"):
        review.export_csv(packet, {"20": choice})
    assert review.export_csv(packet, {"604": {"decision": "defer", "reviewer": "Fixture QA"}})
    with pytest.raises(ValueError, match="Unknown"):
        review.export_csv(packet, {"999": {}})


@pytest.fixture
def local_review_server(tmp_path, monkeypatch, packet):
    monkeypatch.setattr(review, "ROOT", tmp_path)
    output, assets = tmp_path / "reviews/fixture", tmp_path / "data/assets"
    output.mkdir(parents=True)
    assets.mkdir(parents=True)
    source = review.TEMPLATE
    data = json.dumps(packet).replace("<", "\\u003c")
    (output / "index.html").write_text((source / "index.html").read_text().replace("__REVIEW_DATA__", data))
    for name in ("viewer.js", "viewer.css"):
        (output / name).write_bytes((source / name).read_bytes())
    for name in ("context.jpg", "crop.jpg"):
        Image.new("RGB", (10, 10)).save(assets / name)
    server = ThreadingHTTPServer(("127.0.0.1", 0), review.review_handler(output, assets, packet))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    yield f"http://127.0.0.1:{server.server_port}", output, packet
    server.shutdown()
    server.server_close()
    worker.join()


def test_server_limits_routes_and_saves_explicit_csv_under_reviews(local_review_server):
    url, output, packet = local_review_server
    assert urlopen(url).status == 200
    for path in ("/.git/config", "/data/assets/../../secret", "/api/save"):
        with pytest.raises(HTTPError) as error:
            urlopen(url + path)
        assert error.value.code == 404
    payload = json.dumps({"packet_sha256": packet["packet_sha256"], "reviews": {}}).encode()
    request = Request(url + "/api/save", data=payload, headers={"Origin": "https://external.invalid"})
    with pytest.raises(HTTPError) as error:
        urlopen(request)
    assert error.value.code == 403 and not (output / "submissions").exists()
    request = Request(url + "/api/save", data=payload, headers={"Origin": url, "Content-Type": "application/json"})
    result = json.loads(urlopen(request).read())
    target = review.ROOT / result["path"]
    assert target.is_relative_to(output / "submissions")
    rows = list(csv.DictReader(io.StringIO(target.read_text(encoding="utf-8-sig"))))
    assert all(r["decision"] == "" for r in rows)
