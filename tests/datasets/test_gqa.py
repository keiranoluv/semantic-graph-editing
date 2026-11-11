import io
import json
import zipfile

import pytest
from PIL import Image

from src.datasets.gqa import (
    GQAAnnotationLoader,
    build_vocabulary,
    image_archive_index,
    load_archive_image,
    make_splits,
    normalize_gqa,
)
from src.features.geometry import GeometryFeatureExtractor
from src.graph.builder import SceneGraphBuilder
from src.knowledge.transe.core import sha256


def annotation():
    return {
        "width": 100,
        "height": 60,
        "objects": {
            "10": {
                "name": "girl",
                "attributes": ["young"],
                "x": -2,
                "y": 0,
                "w": 20,
                "h": 40,
                "relations": [{"object": "20", "name": p} for p in ["has", "has", "with"]],
            },
            "20": {"name": "hair", "attributes": [], "x": 10, "y": 0, "w": 10, "h": 10, "relations": []},
        },
    }


def test_train_only_vocabulary_no_heldout_leakage():
    rows = {"a": annotation(), "b": annotation()}
    original = build_vocabulary(rows, ["a"])
    rows["b"]["objects"]["10"]["name"] = "heldout_unique"
    assert build_vocabulary(rows, ["a"]) == original
    record, counts = normalize_gqa("b", rows["b"], original)
    assert len(record.objects) == 1 and len(record.relations) == 0
    assert counts["dropped_oov_objects"] == 1 and counts["dropped_relations_filtered_endpoint"] == 3


def test_preserves_duplicate_and_multi_gt_with_label_independent_topology():
    raw = annotation()
    vocab = build_vocabulary({"a": raw}, ["a"])
    record, counts = normalize_gqa("a", raw, vocab)
    assert record.objects[0].object_id == 10 and record.objects[0].bbox_xyxy == (0, 0, 18, 40)
    assert counts["clipped_objects"] == 1
    graph = SceneGraphBuilder().build(record, GeometryFeatureExtractor().extract(record))
    assert graph.edge_label_index.shape == (2, 2)
    assert graph.gt_predicate.numel() == 3
    assert graph.edge_label.tolist() == [-1, 0]


def test_invalid_boxes_remove_incident_edges_and_count_oov_predicates():
    raw = annotation()
    vocab = build_vocabulary({"a": raw}, ["a"])
    raw["objects"]["20"]["w"] = 0
    record, counts = normalize_gqa("a", raw, vocab)
    assert len(record.objects) == 1 and counts["dropped_invalid_box_objects"] == 1
    assert counts["dropped_relations_filtered_endpoint"] == 3
    raw = annotation()
    raw["objects"]["10"]["relations"].append({"object": "20", "name": "unseen"})
    _, counts = normalize_gqa("a", raw, vocab)
    assert counts["dropped_oov_relations"] == 1


def test_raw_dangling_edge_rejected():
    raw = annotation()
    vocab = build_vocabulary({"a": raw}, ["a"])
    raw["objects"]["10"]["relations"][0]["object"] = "999"
    with pytest.raises(ValueError, match="Dangling"):
        normalize_gqa("a", raw, vocab)


def test_split_deterministic_official_boundary_kept():
    train = [str(i) for i in range(10)]
    s = make_splits(train, ["11"], val_size=2)
    assert s == make_splits(reversed(train), ["11"], val_size=2)
    assert s["test"] == ["11"] and len(s["train"]) == 8 and not set(s["train"]) & set(s["val"])
    with pytest.raises(ValueError, match="Overlapping"):
        make_splits(train, ["1"], val_size=2)


def test_zip_image_read_without_extraction(tmp_path):
    archive = tmp_path / "images.zip"
    data = io.BytesIO()
    Image.new("RGB", (10, 20)).save(data, format="JPEG")
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("images/123.jpg", data.getvalue())
    index = image_archive_index(archive)
    assert load_archive_image(archive, "123", index).size == (10, 20)
    assert not (tmp_path / "images").exists()


def test_loader_validates_frozen_sources_and_official_split_membership(tmp_path):
    folder = tmp_path / "sceneGraphs"
    folder.mkdir()
    train = {"a": annotation(), "b": annotation()}
    test = {"c": annotation()}
    paths = [folder / "train_sceneGraphs.json", folder / "val_sceneGraphs.json"]
    for path, data in zip(paths, (train, test), strict=True):
        path.write_text(json.dumps(data))
    manifest = {
        "splits": {"train": ["a"], "val": ["b"], "test": ["c"]},
        "annotation_sha256": {p.name: sha256(p) for p in paths},
        "vocabulary": build_vocabulary(train, ["a"]),
    }
    output = tmp_path / "manifest.json"
    output.write_text(json.dumps(manifest))
    loader = GQAAnnotationLoader(tmp_path, "test", output)
    assert len(loader) == 1 and loader.load(0).image_id == "c"
    paths[1].write_text("{}")
    with pytest.raises(ValueError, match="frozen"):
        GQAAnnotationLoader(tmp_path, "test", output)
