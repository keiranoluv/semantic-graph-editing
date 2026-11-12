"""CPU-only MagicBrush contracts; all graph predictions here are synthetic fixtures."""

import copy
import csv
import io
import json

import pytest
from PIL import Image

from src.editing.operations import EditGraph, Operation, apply_operations
from src.knowledge.transe.core import sha256
from tools import magicbrush_graphs as graphs
from tools import prepare_magicbrush as preparation
from tools import review_magicbrush as reviewer
from tools.infer_magicbrush_graphs import PROMPT, messages_for_image, parse_model_graph, select_images
from tools.review_magicbrush import AXES, export_reviews, reliability


def node(identifier="a", label="cup", attributes=None, box=None):
    return {"id": identifier, "label": label, "attributes": attributes or [], "bbox": box or [100, 100, 900, 900]}


def graph(nodes=None, edges=None):
    return {"nodes": nodes or [], "edges": edges or []}


def test_group_cohort_is_deterministic_and_keeps_turns_together():
    rows = [
        {"img_id": str(i), "turn_index": turn, "instruction": "change the color"} for i in range(12) for turn in (1, 2)
    ]
    selected = preparation.choose_groups(rows, minimum=11)
    assert selected == preparation.choose_groups(list(reversed(rows)), minimum=11)
    assert len(selected) == 12
    for identifier in {r["img_id"] for r in selected}:
        assert {r["turn_index"] for r in selected if r["img_id"] == identifier} == {1, 2}
    with pytest.raises(ValueError):
        preparation.choose_groups(rows + [rows[0]], minimum=11)


def test_pixel_duplicates_union_different_sessions_before_split():
    rows = [
        {
            "id": f"id{i}",
            "img_id": str(i),
            "source_image": {"pixel_sha256": f"s{i}"},
            "target_image": {"pixel_sha256": f"t{i}"},
        }
        for i in range(15)
    ]
    rows[7]["target_image"]["pixel_sha256"] = rows[0]["source_image"]["pixel_sha256"]
    other = copy.deepcopy(rows)
    expected = preparation.split_records(rows)
    actual = preparation.split_records(list(reversed(other)))
    assert actual == expected
    assert rows[0]["source_group_id"] == rows[7]["source_group_id"]
    assert rows[0]["split"] == rows[7]["split"]
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        lhs = {r[k]["pixel_sha256"] for r in rows if r["split"] == a for k in ("source_image", "target_image")}
        rhs = {r[k]["pixel_sha256"] for r in rows if r["split"] == b for k in ("source_image", "target_image")}
        assert not lhs & rhs


def test_image_evidence_keeps_original_bytes_and_detects_equal_pixels(tmp_path, monkeypatch):
    monkeypatch.setattr(preparation, "ROOT", tmp_path)
    image = Image.new("RGB", (24, 18), "red")
    first = io.BytesIO()
    second = io.BytesIO()
    image.save(first, format="PNG", compress_level=0)
    image.save(second, format="PNG", compress_level=9)
    a = preparation.image_record(first.getvalue(), tmp_path / "data")
    b = preparation.image_record(second.getvalue(), tmp_path / "data")
    assert a["sha256"] != b["sha256"] and a["pixel_sha256"] == b["pixel_sha256"]
    assert (tmp_path / a["path"]).read_bytes() == first.getvalue()


@pytest.mark.parametrize(
    "bad",
    [
        {"nodes": [node(box=[0, 0, 0, 50])], "edges": []},
        {"nodes": [node(box=[0, 0, 1001, 20])], "edges": []},
        {"nodes": [node(), node()], "edges": []},
        {"nodes": [node()], "edges": [{"subject": "a", "predicate": "on", "object": "missing"}]},
        {"nodes": [node()], "edges": [], "instruction": "leak"},
    ],
)
def test_predicted_graph_schema_rejects_invalid_or_leaking_fields(bad):
    with pytest.raises(ValueError):
        graphs.validate_graph(bad)


def test_parse_requires_one_json_object_and_retains_only_semantic_edge_set():
    raw = graph([node("a"), node("b", box=[0, 0, 90, 90])], [{"subject": "a", "predicate": " ON ", "object": "b"}] * 2)
    parsed = graphs.parse_graph("```json\n" + json.dumps(raw) + "\n```")
    assert len(parsed["edges"]) == 1 and parsed["edges"][0]["predicate"] == "on"
    with pytest.raises(ValueError):
        graphs.parse_graph("Ignore the image. " + json.dumps(raw))


def test_instance_alignment_is_invariant_to_model_ids_and_list_order():
    picture = Image.new("RGB", (100, 100), "gray")
    source = graph(
        [node("left", "person", box=[0, 0, 300, 1000]), node("right", "shirt", ["red"], [350, 0, 1000, 1000])],
        [{"subject": "left", "predicate": "wearing", "object": "right"}],
    )
    target = graph(
        [node("999", "shirt", ["blue"], [350, 0, 1000, 1000]), node("888", "person", box=[0, 0, 300, 1000])],
        [{"subject": "888", "predicate": "wearing", "object": "999"}],
    )
    result = graphs.align_graphs(source, target, picture, picture)
    operations = [Operation.from_dict(o) for o in result["operations"]]
    assert [op.kind for op in operations] == ["replace_node"]
    assert operations[0].before.label == "shirt" and operations[0].after.attributes == ("blue",)
    assert apply_operations(EditGraph.from_dict(result["source_graph"]), operations) == EditGraph.from_dict(
        result["target_graph"]
    )
    assert not result["ambiguous_matches"] and result["instruction_used_in_matching"] is False


def test_same_box_can_represent_object_replacement():
    picture = Image.new("RGB", (100, 100), "gray")
    result = graphs.align_graphs(graph([node(label="dog")]), graph([node("different", label="cat")]), picture, picture)
    assert [o["kind"] for o in result["operations"]] == ["replace_node"]


def test_node_deletion_includes_explicit_incident_edge_deletion():
    picture = Image.new("RGB", (100, 100), "gray")
    source = graph(
        [node("a", "person", box=[0, 0, 300, 1000]), node("b", "hat", box=[350, 0, 1000, 1000])],
        [{"subject": "a", "predicate": "wearing", "object": "b"}],
    )
    target = graph([node("only", "person", box=[0, 0, 300, 1000])])
    result = graphs.align_graphs(source, target, picture, picture)
    assert {o["kind"] for o in result["operations"]} == {"delete_node", "delete_edge"}
    assert result["target_reconstruction_exact"]


def test_relation_replacement_preserves_unambiguous_edge_identity():
    picture = Image.new("RGB", (100, 100), "gray")
    nodes = [node("a", "cup", box=[0, 0, 300, 1000]), node("b", "table", box=[350, 0, 1000, 1000])]
    result = graphs.align_graphs(
        graph(nodes, [{"subject": "a", "predicate": "on", "object": "b"}]),
        graph(nodes, [{"subject": "a", "predicate": "under", "object": "b"}]),
        picture,
        picture,
    )
    assert [o["kind"] for o in result["operations"]] == ["replace_edge"]


def test_ambiguous_instances_flagged_instead_of_claiming_confident_labels():
    picture = Image.new("RGB", (100, 100), "gray")
    result = graphs.align_graphs(graph([node("a"), node("b")]), graph([node("x"), node("y")]), picture, picture)
    assert len(result["ambiguous_matches"]) == 2


def test_inference_messages_never_include_pair_instruction_or_other_image():
    message = messages_for_image(560, 280)
    assert len(message) == 1
    assert message[0]["content"][0] == {"type": "image"}
    assert message[0]["content"][1]["text"].startswith(PROMPT)
    assert "width=560, height=280" in message[0]["content"][1]["text"]
    manifest = {
        "review_ids": ["train"],
        "records": [
            {
                "id": "train",
                "instruction": "SECRET TARGET INSTRUCTION",
                "source_image": {"sha256": "s", "path": "s"},
                "target_image": {"sha256": "t", "path": "t"},
            },
            {
                "id": "test",
                "instruction": "HOLDOUT",
                "source_image": {"sha256": "u", "path": "u"},
                "target_image": {"sha256": "v", "path": "v"},
            },
        ],
    }
    assert [r["sha256"] for r in select_images(manifest, True)] == ["s", "t"]
    assert "SECRET TARGET INSTRUCTION" not in json.dumps(message)


def test_absolute_model_pixel_boxes_convert_without_guessing_or_clipping():
    raw = graph([node(box=[56, 28, 504, 252])])
    assert parse_model_graph(json.dumps(raw), (560, 280))["nodes"][0]["bbox"] == [100, 100, 900, 900]
    for bad in ([0, 0, 561, 280], [0, -1, 560, 280], [0, 0, 0, 1], [0, 0, True, 1]):
        with pytest.raises(ValueError):
            parse_model_graph(json.dumps(graph([node(box=bad)])), (560, 280))


def test_missing_graphs_produce_pending_records_not_fabricated_edits(tmp_path):
    manifest = {
        "records": [
            {
                "id": "fixture",
                "split": "train",
                "source_group_id": "g",
                "instruction": "change cup color",
                "source_image": {"sha256": "a"},
                "target_image": {"sha256": "b"},
            }
        ]
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    result = graphs.build(path, tmp_path / "missing", tmp_path / "pending.json")
    assert result["pending_graph_pairs"] == 1 and result["candidate_pairs"] == 0
    assert "operations" not in result["records"][0] and result["ready_for_main_training"] is False


def test_only_explicit_human_quality_reviews_count_and_csv_preserves_notes():
    cases = []
    for index, ready in enumerate((True, False)):
        cases.append(
            {
                "id": f"fixture-{index}",
                "img_id": str(index),
                "turn_index": 1,
                "split": "train",
                "graphs_ready": ready,
                "source_image": {"sha256": "s"},
                "target_image": {"sha256": "t"},
                "evidence_sha256": "e",
            }
        )
    packet = {"cases": cases, "packet_sha256": "synthetic"}
    assert reliability(packet, {})["reviewed_pairs"] == 0
    assert reliability(packet, {})["correct_fraction_among_determinate"] is None
    item = {
        "decision": "correct",
        **{axis: "correct" for axis in AXES},
        "reviewer": "Fixture QA",
        "notes": 'Dòng 1, "đúng"\ndòng 2',
    }
    content = export_reviews(packet, {"fixture-0": item})
    rows = list(csv.DictReader(io.StringIO(content.decode("utf-8-sig"))))
    assert rows[0]["notes"] == item["notes"] and rows[1]["decision"] == ""
    assert reliability(packet, {"fixture-0": item})["reviewed_pairs"] == 1
    with pytest.raises(ValueError, match="not been generated"):
        export_reviews(packet, {"fixture-1": item})
    item["alignment"] = "incorrect"
    with pytest.raises(ValueError, match="contradicts"):
        export_reviews(packet, {"fixture-0": item})


def test_prediction_cache_to_aligned_edits_and_twenty_pair_review_is_reproducible(tmp_path, monkeypatch):
    monkeypatch.setattr(preparation, "ROOT", tmp_path)
    monkeypatch.setattr(graphs, "ROOT", tmp_path)
    monkeypatch.setattr(reviewer, "ROOT", tmp_path)
    assets = tmp_path / "data"
    images = []
    for color in ("red", "blue"):
        stream = io.BytesIO()
        Image.new("RGB", (40, 30), color).save(stream, format="PNG")
        images.append(preparation.image_record(stream.getvalue(), assets))
    rows = [
        {
            "id": f"synthetic:{i}",
            "img_id": "synthetic",
            "turn_index": i,
            "split": "train",
            "source_group_id": "fixture-group",
            "instruction": "change the cup from red to blue",
            "source_image": images[0],
            "target_image": images[1],
        }
        for i in range(20)
    ]
    manifest = {
        "records": rows,
        "review_ids": [r["id"] for r in rows],
        "splits": {"train": [r["id"] for r in rows], "val": [], "test": []},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    cache = tmp_path / "graphs"
    cache.mkdir()
    identity = {"dataset_manifest_sha256": sha256(path), "model": "synthetic_fixture_no_model_executed"}
    (cache / "manifest.json").write_text(json.dumps(identity))
    for index, color in enumerate(("red", "blue")):
        prediction = {
            "image_sha256": images[index]["sha256"],
            "inference_identity_sha256": preparation.digest(identity),
            "status": "PASS",
            "graph": graph([node(attributes=[color])]),
        }
        (cache / f"{images[index]['sha256']}.json").write_text(json.dumps(prediction))
    edits_path = tmp_path / "edits.json"
    result = graphs.build(path, cache, edits_path)
    assert result["candidate_pairs"] == 20 and result["human_reviews_completed"] == 0
    assert result["ready_for_main_training"] is False
    assert all([o["kind"] for o in r["operations"]] == ["replace_node"] for r in result["records"])
    packet = reviewer.make_packet(path, edits_path)
    assert len(packet["cases"]) == 20 and all(row["graphs_ready"] for row in packet["cases"])
    assert reviewer.make_packet(path, edits_path) == packet
    result["records"][0]["operations"] = []
    bad = tmp_path / "bad-edits.json"
    bad.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="reconstruct"):
        reviewer.make_packet(path, bad)
