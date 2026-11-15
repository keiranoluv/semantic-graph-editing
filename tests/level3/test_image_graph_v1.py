"""CPU-only contracts; fake outputs are test fixtures, never benchmark results."""

import copy
import json
import os
import subprocess
import sys

import pytest
import torch
from PIL import Image

from src.editing.operations import EditGraph, Node
from tools.level3_image_graph_v1 import backend, runner
from tools.level3_image_graph_v1.compatibility import audit_graph, load_task_b
from tools.level3_image_graph_v1.core import (
    DEFAULT_CONFIG,
    digest,
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
from tools.level3_image_graph_v1.review import AXES, apply_review, graph_svg, render_review


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


def submission(record):
    return {
        "schema": "level3-image-graph-review-v1",
        "reviews": [
            {
                "candidate_id": record["candidate_id"],
                "base_record_sha256": digest(record),
                "decision": "accept",
                "reviewer": "CPU-test-fixture",
                "notes": "fixture only",
                "checks": {a: True for a in AXES},
                "source_graph": copy.deepcopy(record["source_graph"]),
                "boxes": {k: v["bbox_xyxy"] for k, v in record["grounding"]["nodes"].items()},
            }
        ],
    }


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


def test_actual_reltr_tensor_contract_background_direction_boxes():
    vocab = {"CLASSES": ["N/A", "person", "bike"], "REL_CLASSES": ["__background__", "riding"]}
    output = {}
    for role, label in (("pred", 3), ("sub", 1), ("obj", 2)):
        logits = torch.full((1, 1, 4), -9.0)
        logits[0, 0, label] = 9
        output[role + "_logits"] = logits
        output[role + "_boxes"] = torch.tensor([[[0.5, 0.5, 0.5, 0.5]]])
    output["rel_logits"] = torch.tensor([[[-9.0, 9.0, -9.0]]])
    raw = backend.decode_outputs(output, (100, 80), vocab)
    assert len(raw["detections"]) == 2
    assert raw["detections"][0]["bbox_xyxy"] == [25, 20, 75, 60]
    assert raw["relations"][0]["subject"] == "sub-000"
    assert raw["relations"][0]["object"] == "obj-000"
    output["rel_logits"] = torch.tensor([[[9.0, -9.0, -9.0]]])
    assert not backend.decode_outputs(output, (100, 80), vocab)["relations"]
    output["pred_logits"][0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        backend.decode_outputs(output, (100, 80), vocab)


def test_attribute_abstention():
    assert backend.select_attributes([0.35, 0.2], ["red", "blue"], 0.2, 0.02) == "red"
    assert backend.select_attributes([0.35, 0.34], ["red", "blue"], 0.2, 0.02) is None
    assert backend.select_attributes([0.15, 0.1], ["red", "blue"], 0.2, 0.02) is None


def test_frozen_region_attribute_inference(record, config):
    # Exercise crop/encode/prompt/evidence wiring with a CPU-only tiny model.
    class FakeClip:
        def encode_image(self, x):
            assert x.device.type == "cpu" and not torch.is_grad_enabled()
            return torch.tensor([[1.0, 0.0]])

        def encode_text(self, tokens):
            result = torch.tensor([[0.0, 1.0]]).repeat(len(tokens), 1)
            result[0] = torch.tensor([1.0, 0.0])
            return result

    graph, evidence = backend.add_attributes(
        FakeClip(),
        lambda im: torch.zeros(3, 8, 8),
        Image.new("RGB", (100, 80)),
        EditGraph.from_dict(record["source_graph"]),
        record["grounding"],
        config,
        "cpu",
    )
    assert all(n.attributes == ("black", "wooden") for n in graph.nodes)
    record["source_graph"] = graph.to_dict()
    validate_record(record)
    assert len(evidence["n001"]) == 19


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


def test_review_correction_and_original_immutable(record):
    before = digest(record)
    data = submission(record)
    r = data["reviews"][0]
    r["source_graph"]["nodes"][0]["attributes"] = ["red"]
    accepted, decisions = apply_review([record], data)
    assert digest(record) == before and decisions[0]["decision"] == "accept"
    corrected = accepted[0]
    assert corrected["human_reviewed"] is True
    assert corrected["grounding"]["nodes"]["n001"]["confidence"] is None
    assert corrected["review"]["operations_from_candidate"][0]["kind"] == "replace_node"
    validate_record(corrected)


@pytest.mark.parametrize(
    "fault", ["stale", "unchecked", "anonymous", "dangling", "box", "duplicate", "empty", "attribute_string"]
)
def test_review_rejects_invalid_evidence(record, fault):
    data = submission(record)
    r = data["reviews"][0]
    if fault == "stale":
        r["base_record_sha256"] = "wrong"
    elif fault == "unchecked":
        r["checks"]["attributes"] = False
    elif fault == "anonymous":
        r["reviewer"] = ""
    elif fault == "dangling":
        r["source_graph"]["nodes"] = r["source_graph"]["nodes"][1:]
    elif fault == "box":
        r["boxes"]["n001"] = [0, 0, 999, 8]
    elif fault == "duplicate":
        data["reviews"].append(r)
    elif fault == "attribute_string":
        r["source_graph"]["nodes"][0]["attributes"] = "red"
    else:
        r["source_graph"] = {"nodes": [], "edges": []}
        r["boxes"] = {}
    with pytest.raises(ValueError):
        apply_review([record], data)


def test_pending_and_rejected_not_exported(record):
    for decision in ("pending", "reject"):
        data = submission(record)
        data["reviews"][0]["decision"] = decision
        assert not apply_review([record], data)[0]


def test_viewer_escape_and_hashed_template(record, tmp_path):
    path = tmp_path / record["image"]["image_path"]
    path.parent.mkdir(parents=True)
    Image.new("RGB", (100, 80)).save(path)
    record["image"]["sha256"] = sha256(path)
    record["source_sha256"] = sha256(path)
    output = tmp_path / "view"
    output.mkdir()
    render_review(output, [record], tmp_path)
    assert "__REVIEW_DATA__" not in (output / "review.html").read_text()
    assert read(output / "review-template.json")["reviews"][0]["base_record_sha256"] == digest(record)
    malicious = EditGraph((Node("n001", "</script><script>alert(1)</script>"),), ())
    assert "<script>" not in graph_svg(malicious)


def test_task_b_not_assumed_compatible(record):
    graph = EditGraph.from_dict(record["source_graph"])
    ckpt = {"config": {"max_nodes": 7, "max_edges": 42}, "vocabulary": ['"person"', '"bike"', '"riding"', "[]"]}
    result = audit_graph(graph, ckpt)
    assert result["codec_roundtrip_valid"] and result["within_task_b_input_budget"]
    assert result["source_payload_oov"] == []
    assert not result["direct_model_use_approved"] and result["one_way_edge_ids"] == ["e001"]
    large = EditGraph(tuple(Node(f"n{i:03d}", "person") for i in range(8)), ())
    assert not audit_graph(large, ckpt)["within_task_b_input_budget"]
    attr = EditGraph((Node("n001", "person", ("red",)),), ())
    assert '["red"]' in audit_graph(attr, ckpt)["source_payload_oov"]


def test_task_b_checkpoint_config_matches_weights(tmp_path):
    path = tmp_path / "checkpoint.pt"
    saved = {
        "vocabulary": ['"person"'],
        "config": {"max_nodes": 7, "max_edges": 42},
        "experiment_identity": {"test_fixture": True},
        "state_dict": {
            "node_identity.weight": torch.zeros(7, 3),
            "edge_identity.weight": torch.zeros(42, 3),
            "output.weight": torch.zeros(5, 3),
        },
    }
    torch.save(saved, path)
    assert load_task_b(path)["config"]["max_nodes"] == 7
    saved["config"]["max_nodes"] = 20
    torch.save(saved, path)
    with pytest.raises(ValueError, match="does_not_match_weights"):
        load_task_b(path)


def test_preflight_refuses_other_cohort_before_loading_model(pilot):
    with pytest.raises(ValueError, match="unexpected_locked_pilot_revision"):
        runner.main(["preflight", "--locked", str(pilot)])


def test_record_image_identity_not_interchangeable(record):
    record["candidate_id"] = "travel-wrong"
    with pytest.raises(ValueError, match="identity_mismatch"):
        validate_record(record)


def test_cli_help_and_cpu_smoke_cannot_request_gpu():
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    result = subprocess.run(
        [sys.executable, "-m", "tools.level3_image_graph_v1", "--help"], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0 and "compatibility" in result.stdout
    result = subprocess.run(
        [sys.executable, "-m", "tools.level3_image_graph_v1", "smoke", "--output", "/tmp/unused", "--device", "cuda"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0


def test_cpu_end_to_end_all30_with_fake_backend(pilot, tmp_path, raw, config, monkeypatch):
    manifest, cohort = load_pilot(pilot)
    before = {str(p): sha256(p) for p in pilot.rglob("*") if p.is_file()}
    events = []
    monkeypatch.setattr(runner, "preflight", lambda a: (config, manifest, cohort, {"test_fixture": True}, {}, None))
    monkeypatch.setattr(runner, "load_reltr", lambda *a: events.append("load_reltr"))
    monkeypatch.setattr(runner, "infer_reltr", lambda *a: events.append("reltr") or raw)
    monkeypatch.setattr(runner, "load_clip", lambda *a: (events.append("load_clip"), None))
    monkeypatch.setattr(runner, "add_attributes", lambda m, p, i, g, grounding, c, d: (g, {}))
    output = tmp_path / "output"
    runner.main(["infer", "--device", "cpu", "--locked", str(pilot), "--output", str(output)])
    bundle = runner.read_bundle(output)
    assert len(bundle["records"]) == 30
    assert read(output / "receipt.json")["pilot_inference_complete"]
    assert events == ["load_reltr"] + ["reltr"] * 30 + ["load_clip"]
    assert before == {str(p): sha256(p) for p in pilot.rglob("*") if p.is_file()}
    monkeypatch.setattr(runner, "load_task_b", lambda *a: None)
    runner.main(
        [
            "review",
            "--locked",
            str(pilot),
            "--bundle",
            str(output),
            "--submission",
            str(output / "review-template.json"),
            "--output",
            str(tmp_path / "reviewed"),
        ]
    )
    reviewed = read(tmp_path / "reviewed/reviewed.json")
    assert reviewed["records"] == []
    assert len(reviewed["decisions"]) == 30
    with pytest.raises(FileExistsError):
        runner.main(["infer", "--device", "cpu", "--locked", str(pilot), "--output", str(output)])
    path = output / "candidates.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="modified"):
        runner.read_bundle(output)
