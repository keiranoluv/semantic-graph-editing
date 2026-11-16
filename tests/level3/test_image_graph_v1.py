"""CPU-only contracts; fake outputs are test fixtures, never benchmark results."""

import json

import pytest
from PIL import Image

from src.editing.operations import EditGraph
from tools.level3_image_graph_v1.core import (
    DEFAULT_CONFIG,
    graph_from_raw,
    load_pilot,
    make_record,
    outside_locked,
    read,
    sha256,
    validate_box,
    validate_record,
    write,
)


@pytest.fixture
def config():
    return read(DEFAULT_CONFIG)


@pytest.fixture
def row():
    return {
        "image_id": "sidewalk-fixture",
        "width": 100,
        "height": 80,
        "sha256": "a" * 64,
        "image_path": "images/sidewalk-fixture.jpg",
        "dataset": "sidewalk",
    }


@pytest.fixture
def raw():
    return {
        "detections": [
            {"id": "entity-000", "label": "person", "confidence": 0.95, "bbox_xyxy": [0, 0, 30, 70]},
            {"id": "sub-000", "label": "person", "confidence": 0.9, "bbox_xyxy": [1, 1, 30, 70]},
            {"id": "obj-000", "label": "bike", "confidence": 0.8, "bbox_xyxy": [25, 30, 90, 75]},
            {"id": "entity-001", "label": "person", "confidence": 0.7, "bbox_xyxy": [70, 0, 95, 25]},
            {"id": "entity-002", "label": "tree", "confidence": 0.1, "bbox_xyxy": [40, 0, 60, 20]},
        ],
        "relations": [
            {
                "id": "rel-000",
                "subject": "sub-000",
                "object": "obj-000",
                "predicate": "riding",
                "confidence": 0.85,
                "triplet_score": 0.612,
            },
            {
                "id": "rel-001",
                "subject": "entity-000",
                "object": "obj-000",
                "predicate": "riding",
                "confidence": 0.75,
                "triplet_score": 0.57,
            },
        ],
    }


@pytest.fixture
def record(raw, row, config):
    graph, grounding = graph_from_raw(raw, row, config)
    return make_record(row, graph, grounding, {"test_fixture": True})


@pytest.fixture
def pilot(tmp_path):
    locked = tmp_path / "locked"
    rows, tasks = [], []
    for group in ("sidewalk", "travel"):
        for i in range(15):
            identifier = f"{group}-{i:02d}"
            path = locked / "images" / (identifier + ".png")
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (100, 80), (i, 30 if group == "sidewalk" else 60, 80)).save(path)
            row = {
                "image_id": identifier,
                "dataset": group,
                "image_path": str(path.relative_to(locked)),
                "sha256": sha256(path),
                "width": 100,
                "height": 80,
            }
            rows.append(row)
            tasks.append(
                {
                    "image_id": identifier,
                    "image_path": row["image_path"],
                    "source_sha256": row["sha256"],
                    "source_graph": None,
                }
            )
    write(locked / "manifest.json", {"image_count": 30, "images": rows})
    (locked / "source_graph_tasks.jsonl").write_text("\n".join(json.dumps(t) for t in tasks))
    return locked


def test_directed_edges_merge_evidence_and_distinct_instances(raw, row, config):
    graph, meta = graph_from_raw(raw, row, config)
    assert len(graph.nodes) == 3 and len(graph.edges) == 1
    assert sum(n.label == "person" for n in graph.nodes) == 2
    edge = graph.edges[0]
    assert (edge.subject, edge.predicate, edge.object) == ("n001", "riding", "n002")
    assert meta["nodes"]["n001"]["raw_detection_ids"] == ["entity-000", "sub-000"]
    assert meta["edges"]["e001"]["raw_relation_ids"] == ["rel-000", "rel-001"]
    assert EditGraph.from_dict(graph.to_dict()) == graph
    raw["detections"].reverse()
    raw["relations"].reverse()
    assert graph_from_raw(raw, row, config) == (graph, meta)


@pytest.mark.parametrize("box", [[-1, 0, 10, 10], [0, 0, 101, 5], [4, 0, 4, 8], [0, 0, float("nan"), 7]])
def test_invalid_boxes(box):
    with pytest.raises(ValueError):
        validate_box(box, 100, 80)


@pytest.mark.parametrize("change", ["nan", "duplicate", "dangling"])
def test_bad_raw_rejected(raw, row, config, change):
    if change == "nan":
        raw["detections"][0]["confidence"] = float("nan")
    elif change == "duplicate":
        raw["detections"].append(raw["detections"][0])
    else:
        raw["relations"][0]["object"] = "absent"
    with pytest.raises(ValueError):
        graph_from_raw(raw, row, config)


def test_pilot_hashes_and_quota(pilot):
    assert load_pilot(pilot)[1]["counts"] == {"sidewalk": 15, "travel": 15}
    path = pilot / "images/sidewalk-00.png"
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash_mismatch"):
        load_pilot(pilot)


def test_pilot_task_identity(pilot):
    path = pilot / "source_graph_tasks.jsonl"
    rows = path.read_text().splitlines()
    task = json.loads(rows[0])
    task["source_sha256"] = "changed"
    rows[0] = json.dumps(task)
    path.write_text("\n".join(rows))
    with pytest.raises(ValueError, match="task_identity"):
        load_pilot(pilot)


def test_output_protection(tmp_path):
    locked = tmp_path / "locked"
    with pytest.raises(ValueError):
        outside_locked(locked / "new.json", locked)
    with pytest.raises(ValueError):
        outside_locked(tmp_path, locked)
    link = tmp_path / "alias"
    locked.mkdir()
    link.symlink_to(locked, target_is_directory=True)
    with pytest.raises(ValueError):
        outside_locked(link / "new.json", locked)


def test_record_image_identity_not_interchangeable(record):
    record["candidate_id"] = "travel-wrong"
    with pytest.raises(ValueError, match="identity_mismatch"):
        validate_record(record)
