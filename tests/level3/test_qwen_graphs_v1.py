import json

import pytest
from PIL import Image

from tools.level3_image_graph_v1.core import sha256, write
from tools.level3_qwen_graphs_v1 import core
from tools.level3_qwen_graphs_v1.viewer import render
from tools.level3_sgg_pretrained_v2.adapter import adapt


def test_white_alpha_matches_review_and_keeps_native(tmp_path):
    native = tmp_path / "native.png"
    im = Image.new("RGBA", (100, 80), (0, 0, 0, 0))
    im.putpixel((1, 1), (255, 0, 0, 128))
    im.save(native)
    before = sha256(native)
    normalized = core.normalize(native, tmp_path / "input.png")
    with Image.open(normalized["image_path"]) as result:
        assert result.mode == "RGB" and result.size == (100, 80)
        assert result.getpixel((0, 0)) == (255, 255, 255)
        assert result.getpixel((1, 1)) == (255, 127, 127)
    assert normalized["native"]["sha256"] == before == sha256(native)
    with pytest.raises(FileExistsError):
        core.normalize(native, tmp_path / "input.png")


def test_exif_transpose_before_graph_coordinates(tmp_path):
    path = tmp_path / "rotated.jpg"
    im = Image.new("RGB", (100, 50), "red")
    exif = im.getexif()
    exif[274] = 6
    im.save(path, exif=exif)
    r = core.normalize(path, tmp_path / "normal.png")
    assert (r["width"], r["height"]) == (50, 100)


def rows(tmp_path):
    source, target = tmp_path / "source.png", tmp_path / "target.png"
    Image.new("RGB", (100, 80), "blue").save(source)
    Image.new("RGBA", (100, 80), "red").save(target)
    return [
        {
            "task_id": "t1",
            "image_id": "i1",
            "status": "ok",
            "source_path": str(source),
            "target_path": str(target),
            "source_sha256": sha256(source),
            "target": {"sha256": sha256(target)},
        }
    ]


def test_pair_hashes_and_status_not_review_labels(tmp_path):
    r = rows(tmp_path)
    r[0]["human_quality_label"] = None
    assert core.validate_rows(r, tmp_path, expected=1) == r
    (tmp_path / "target.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="pair_image_changed"):
        core.validate_rows(r, tmp_path, expected=1)


def test_missing_and_duplicate_pairs_rejected(tmp_path):
    r = rows(tmp_path)
    with pytest.raises(ValueError, match="expected_unique"):
        core.validate_rows(r, tmp_path)
    with pytest.raises(ValueError, match="expected_unique"):
        core.validate_rows(r + r, tmp_path, expected=2)
    r[0]["status"] = "error"
    with pytest.raises(ValueError, match="invalid_pair"):
        core.validate_rows(r, tmp_path, expected=1)


def test_output_is_new_and_outside_old_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "RUNS", tmp_path / "new")
    out = core.new_output(tmp_path / "new/run1")
    with pytest.raises(FileExistsError):
        core.new_output(out)
    with pytest.raises(ValueError):
        core.new_output(tmp_path / "old")


def test_directed_relations_local_ids_and_missing_attribute_head():
    row = {"image_id": "t--source", "sha256": "s", "width": 100, "height": 80}
    objects = [
        {"label": "person", "bbox_xyxy": [0, 0, 20, 40], "confidence": 0.8},
        {"label": "chair", "bbox_xyxy": [20, 0, 50, 40], "confidence": 0.9},
    ]
    rec = adapt(
        row, "react++", objects, [{"subject": 0, "object": 1, "predicate": "sitting on", "confidence": 0.7}], {}, {}
    )
    edge = rec["source_graph"]["edges"][0]
    assert (edge["subject"], edge["object"]) == ("n000", "n001")
    assert rec["grounding"]["edges"][edge["id"]]["confidence"] == 0.7
    assert rec["attribute_coverage"] == "unsupported_no_attribute_head"
    assert rec["human_reviewed"] is False
    assert rec["identity_scope"] == "per_image_only_not_cross_image_correspondence"


def test_viewer_preserves_empty_prediction_vs_parse_error_and_escapes(tmp_path):
    image = tmp_path / "i.png"
    Image.new("RGB", (100, 80), "red").save(image)
    row = {"image_id": "s", "sha256": sha256(image), "width": 100, "height": 80}
    rec = adapt(row, "react++", [], [], {}, {})
    write(tmp_path / "source.json", rec)
    write(tmp_path / "target.json", {"status": "parse_error", "image": row, "error": "bad output"})
    roles = {
        role: {
            "native_image_path": str(image),
            "graph_path": str(tmp_path / (role + ".json")),
            "status": "ok" if role == "source" else "parse_error",
        }
        for role in ("source", "target")
    }
    render(
        [{"task_id": "t", "image_id": "i", "dataset": "x", "prompt": "<script>bad()</script>", "roles": roles}],
        tmp_path,
    )
    text = (tmp_path / "comparison.html").read_text()
    assert "<script>bad()</script>" not in text
    data = json.loads(text.split('<script id="data" type="application/json">')[1].split("</script>")[0])
    assert data[0]["roles"]["source"]["graph"] == {"nodes": [], "edges": []}
    assert data[0]["roles"]["target"]["graph"] is None
    assert data[0]["roles"]["target"]["status"] == "parse_error"
