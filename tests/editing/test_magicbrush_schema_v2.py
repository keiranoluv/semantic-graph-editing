"""CPU schema/gate fixtures only; no claim of improved real v2 inference yet."""

import json

import pytest
import torch

from src.knowledge.transe.core import sha256
from tools import infer_magicbrush_graphs_v2 as inference
from tools import magicbrush_review_gate_v2 as gate
from tools.prepare_magicbrush import digest, encode


def raw_node(identifier="n1", label="cup", box=None):
    return {"id": identifier, "label": label, "attributes": [], "bbox": box or [10, 20, 90, 80]}


@pytest.mark.parametrize("mode", ["flat", "nested", "alias", "alias_nested"])
def test_only_lossless_box_formats_are_normalized(mode):
    node = raw_node()
    if "nested" in mode:
        node["bbox"] = [node["bbox"]]
    if "alias" in mode:
        node["bbox_2d"] = node.pop("bbox")
    graph, changes = inference.parse_model_graph_v2(json.dumps({"nodes": [node], "edges": []}), (100, 100))
    assert graph["nodes"][0]["bbox"] == [100, 200, 900, 800]
    assert len(changes) == ({"flat": 0, "nested": 1, "alias": 1, "alias_nested": 2}[mode])


def test_dangling_label_attribute_and_missing_node_endpoints_remain_errors():
    for endpoint in ("plate", "red", "missing", "n1"):
        raw = {
            "nodes": [raw_node(), raw_node("n2", "plate")],
            "edges": [{"subject": "n1", "predicate": "on", "object": endpoint}],
        }
        with pytest.raises(ValueError, match="edge endpoint"):
            inference.parse_model_graph_v2(json.dumps(raw), (100, 100))


def test_conflicting_or_nonpositive_boxes_are_not_guessed_or_clipped():
    node = raw_node()
    node["bbox_2d"] = [0, 0, 20, 20]
    with pytest.raises(ValueError, match="Conflicting"):
        inference.parse_model_graph_v2(json.dumps({"nodes": [node], "edges": []}), (100, 100))
    for bad in ([0, 0, 101, 50], [10, 10, 5, 90], [[1, 2], [3, 4]], [True, 0, 50, 50]):
        with pytest.raises(ValueError):
            inference.parse_model_graph_v2(json.dumps({"nodes": [raw_node(box=bad)], "edges": []}), (100, 100))


def test_v2_prompt_uses_only_one_image_and_declares_exact_id_constraints():
    messages = inference.messages_for_image(616, 504)
    assert len(messages) == 1 and messages[0]["content"][0] == {"type": "image"}
    prompt = messages[0]["content"][1]["text"]
    assert "width=616, height=504" in prompt
    assert "copied EXACTLY" in prompt and "ONE FLAT list" in prompt


def test_scope_is_explicit_and_review_only_plan_never_queries_cuda(tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU plan touched CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    manifest = {
        "review_ids": [str(i) for i in range(20)],
        "records": [
            {
                "id": str(i),
                "instruction": "SECRET EDIT NOT IN INPUT",
                "source_image": {"sha256": f"s{i}"},
                "target_image": {"sha256": f"t{i}"},
            }
            for i in range(20)
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_bytes(encode(manifest))
    inference.main(["--review-only", "--plan", "--manifest", str(path)])
    assert json.loads(capsys.readouterr().out)["images"] == 40
    with pytest.raises(SystemExit):
        inference.main(["--plan"])


def test_full_scope_gate_blocks_before_cuda_even_in_plan(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Blocked full touched CUDA")

    def blocked(*args, **kwargs):
        raise ValueError("fixture human review missing")

    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(gate, "verify_gate", blocked)
    with pytest.raises(ValueError, match="human review missing"):
        inference.main(["--full", "--plan"])


def test_selector_will_not_create_reviews_when_no_human_csv_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    config = {
        "review_csv": "reviews/fixture/selected/review.csv",
        "review_packet": "reviews/fixture/selected/packet.json",
    }
    with pytest.raises(ValueError, match="No human CSV"):
        gate.select_review(config, review_dir=tmp_path / "reviews/fixture")
    assert not (tmp_path / "reviews/fixture/selected/review.csv").exists()


def test_status_counts_only_frozen_pilot_images_and_rejects_cache_tampering(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    manifest = {
        "review_ids": ["a"],
        "records": [
            {"id": "a", "source_image": {"sha256": "s"}, "target_image": {"sha256": "t"}},
            {"id": "holdout", "source_image": {"sha256": "u"}, "target_image": {"sha256": "v"}},
        ],
    }
    manifest_path = tmp_path / "dataset.json"
    manifest_path.write_bytes(encode(manifest))
    config = {"output": "data/cache"}
    folder = tmp_path / config["output"]
    folder.mkdir(parents=True)
    cache = {"dataset_manifest_sha256": sha256(manifest_path), "config": config, "source_sha256": {}}
    (folder / "manifest.json").write_bytes(encode(cache))
    for image, status in (("s", "PASS"), ("t", "ERROR")):
        (folder / f"{image}.json").write_bytes(
            encode(
                {
                    "image_sha256": image,
                    "inference_identity_sha256": digest(cache),
                    "status": status,
                    "error": "fixture invalid box",
                }
            )
        )
    (folder / "u.json").write_text("Not a valid JSON; not selected and must not be opened")
    status = gate.pilot_status(config, manifest_path)
    assert status["expected_images"] == 2 and status["image_counts"] == {"PASS": 1, "ERROR": 1}
    assert not status["all_pilot_image_graphs_structurally_valid"]
    (folder / "s.json").write_bytes(
        encode({"image_sha256": "wrong", "inference_identity_sha256": digest(cache), "status": "PASS"})
    )
    with pytest.raises(ValueError, match="provenance mismatch"):
        gate.pilot_status(config, manifest_path)


def test_selector_copies_latest_actual_human_bytes_and_never_falls_back(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    review = tmp_path / "reviews/fixture"
    (review / "submissions").mkdir(parents=True)
    (review / "packets").mkdir()
    packet_hash = "a" * 64
    source = review / "submissions/review-actual.csv"
    payload = (
        b"\xef\xbb\xbfreview_packet_sha256,decision,notes\r\n"
        + packet_hash.encode()
        + b',correct,"actual HUMAN fixture"\r\n'
    )
    source.write_bytes(payload)
    (review / "packets" / f"{packet_hash}.json").write_bytes(b'{"fixture": true}\n')
    calls = []

    def verified(config, csv, packet, manifest):
        calls.append(csv)
        return {"ready": True, "automatically_approved": False}

    monkeypatch.setattr(gate, "verify_gate", verified)
    config = {
        "review_csv": "reviews/fixture/selected/review.csv",
        "review_packet": "reviews/fixture/selected/packet.json",
    }
    result = gate.select_review(config, review_dir=review)
    assert calls == [source] and (tmp_path / config["review_csv"]).read_bytes() == payload
    assert (tmp_path / config["review_packet"]).read_bytes() == b'{"fixture": true}\n'
    assert result["human_decisions_created"] == 0
    newest = review / "submissions/review-new.csv"
    newest.write_bytes(payload)
    source_mtime = source.stat().st_mtime_ns
    import os

    os.utime(newest, ns=(source_mtime + 1_000_000, source_mtime + 1_000_000))

    def rejected(config, csv, packet, manifest):
        assert csv == newest
        raise ValueError("newest human CSV is not ready")

    monkeypatch.setattr(gate, "verify_gate", rejected)
    with pytest.raises(ValueError, match="newest human"):
        gate.select_review(config, review_dir=review)
