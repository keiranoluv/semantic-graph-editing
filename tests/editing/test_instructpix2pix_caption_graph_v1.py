"""CPU grammar, provenance, streaming, review and predictor integration tests.

All generated records in this file are test fixtures, never dataset results.
"""

import csv
import hashlib
import io
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from src.editing.evaluation import evaluate_edits
from src.editing.operations import Edge, EditGraph, Node, Operation, apply_operations
from src.knowledge.transe.core import sha256
from tools.edit_model_v1.codec import decode_actions, encode_actions
from tools.edit_model_v1.data import feature_texts
from tools.instructpix2pix_caption_graph_v1 import dataset, review, runner
from tools.instructpix2pix_caption_graph_v1.alignment import align
from tools.instructpix2pix_caption_graph_v1.compiler import compile_record
from tools.instructpix2pix_caption_graph_v1.instruction_check import intent
from tools.instructpix2pix_caption_graph_v1.parser import parse
from tools.instructpix2pix_caption_graph_v1.report import summarize
from tools.instructpix2pix_caption_graph_v1.validator import validate
from tools.magicbrush_grounded_v1.language import tagger
from tools.magicbrush_joint_v1.storage import read, write_once
from tools.prepare_magicbrush import digest
from tools.review_magicbrush import AXES


@pytest.fixture(autouse=True)
def pinned_cpu_tagger():
    # Match the repository's existing NLP tests; downloading resources belongs
    # to prepare, never to pytest or a GPU task.
    try:
        tagger()
    except FileNotFoundError:
        pytest.skip("run CPU prepare to install the hash-pinned English POS tagger")


def row(
    source="A red car beside a house.",
    target="A blue car beside a house.",
    instruction="Make the car blue.",
    identifier="fixture",
):
    return {
        "id": identifier,
        "original_prompt": source,
        "edited_prompt": target,
        "edit_prompt": instruction,
        "upstream_image_paths": {"original_image": identifier + ".jpg"},
        "provenance": {"fixture": True},
    }


SUPPORTED = [
    (
        "A red car parked beside a house.",
        "A blue car parked beside a house.",
        "Change the car's color to blue.",
        ["replace_node"],
    ),
    ("A cat sitting on a sofa.", "A dog sitting on a sofa.", "Replace the cat with a dog.", ["replace_node"]),
    ("A cup on the table.", "A cup under the table.", "Move the cup under the table.", ["replace_edge"]),
    (
        "A sofa in a living room.",
        "A sofa with a pillow on it in a living room.",
        "Add a pillow on the sofa.",
        ["add_node", "add_edge"],
    ),
    (
        "A sofa with a pillow on it in a living room.",
        "A sofa in a living room.",
        "Remove the pillow.",
        ["delete_node", "delete_edge"],
    ),
    ("A sofa.", "A sofa and a pillow.", "Add a pillow.", ["add_node"]),
    ("A cup and a table.", "A cup on the table.", "Put the cup on the table.", ["add_edge"]),
    ("A cup on a table.", "A cup and a table.", "Take the cup off the table.", ["delete_edge"]),
    (
        "A small black dog behind a wooden fence.",
        "A small white dog behind a wooden fence.",
        "Paint the black dog white.",
        ["replace_node"],
    ),
    ("A lamp to the left of a chair.", "A lamp above a chair.", "Move the lamp above the chair.", ["replace_edge"]),
]


@pytest.mark.parametrize("source,target,instruction,kinds", SUPPORTED)
def test_real_grammar_and_transactions(source, target, instruction, kinds):
    result = compile_record(row(source, target, instruction))
    assert result["status"] == "candidate", result["problems"]
    assert [o["kind"] for o in result["operations"]] == kinds
    assert result["visual_verification_status"] == "unverified"
    assert result["human_review_status"] == "pending"
    a, b = (EditGraph.from_dict(result[k]) for k in ("source_graph", "target_graph"))
    operations = tuple(Operation.from_dict(o) for o in result["operations"])
    assert apply_operations(a, operations) == b
    assert apply_operations(a, decode_actions(encode_actions(operations)[1:], a)) == b
    metrics = evaluate_edits([{"source": a, "target": b, "expected": operations, "predicted": operations}])
    assert metrics["exact_target_graph_accuracy"] == 1


def test_object_attributes_relations_and_full_word_consumption():
    parsed = parse("A small red car parked beside a wooden house.")
    assert [(n.label, n.attributes) for n in parsed.graph.nodes] == [("car", ("red", "small")), ("house", ("wooden",))]
    assert parsed.graph.edges == (Edge("e001", "n001", "beside", "n002"),)
    assert parsed.evidence["posture"] == "parked"
    assert parse("A car is blue.").graph.nodes == (Node("n001", "car", ("blue",)),)
    with pytest.raises(ValueError):
        parse("A mysterious red car accelerating beside a house.")


@pytest.mark.parametrize(
    "caption,reason",
    [
        ("A cat beside a cat.", "multiple_indistinguishable_instances"),
        ("Two cats on a sofa.", "unsupported"),
        ("A cat on it.", "unresolved_pronoun"),
        ("A man and his dog.", "unresolved_pronoun"),
        ("An oil painting of a red car.", "style_global"),
        ("A cartoon dog.", "style_global"),
        ("A man chasing a dog.", "unsupported"),
        ("A cup on a table beside a chair.", "ambiguous_relation_attachment"),
        ("A red and blue car.", "unsupported"),
        ("A forest, sunset, beautiful.", "unsupported"),
        ("A cat that is on a sofa.", "unresolved_pronoun"),
    ],
)
def test_explicitly_unsupported_captions(caption, reason):
    with pytest.raises(ValueError, match=reason):
        parse(caption)


@pytest.mark.parametrize(
    "source,target,instruction",
    [
        ("A red car beside a house.", "A blue car beside a house.", "Remove a tree."),
        ("A red car beside a house.", "A blue car beside a house.", "Make the house blue."),
        ("A red car beside a house.", "A shiny blue car beside a house.", "Make the car blue."),
        ("A cat on a sofa.", "A dog under a sofa.", "Replace the cat with a dog."),
        ("A cat sitting on a sofa.", "A dog standing on a sofa.", "Replace the cat with a dog."),
        ("A sofa.", "A pillow on a sofa.", "Add a pillow."),
        ("A cat on a sofa.", "A cat on a sofa.", "Remove the cat."),
        ("A cup on a table.", "A table on a cup.", "Move the cup under the table."),
    ],
)
def test_disagreement_or_graph_drift_rejected(source, target, instruction):
    result = compile_record(row(source, target, instruction))
    assert result["status"] == "rejected"
    assert result["problems"]


def test_alignment_is_not_serialized_id_matching():
    a, b = parse("A sofa and a cat."), parse("A dog and a sofa.")
    source, target, _ = align(a, b, intent("Replace the cat with a dog."))
    assert {n.label: n.id for n in source.nodes}["sofa"] == {n.label: n.id for n in target.nodes}["sofa"]
    assert {n.label: n.id for n in source.nodes}["cat"] == {n.label: n.id for n in target.nodes}["dog"]
    with pytest.raises(ValueError, match="ambiguous"):
        align(parse("A cat and a dog."), parse("A bird and a horse."), intent("Replace the cat with a bird."))


def test_source_ids_independent_of_target_and_instruction():
    a = compile_record(row())
    b = compile_record(row(target="A green car beside a house.", instruction="Make the car green."))
    assert a["source_graph"] == b["source_graph"]
    assert compile_record(row()) == a


@pytest.mark.parametrize("broken", [None, "", 42, [], {}])
def test_malformed_examples(broken):
    r = row()
    r["original_prompt"] = broken
    result = compile_record(r)
    assert not result["stages"]["fields"]
    assert result["status"] == "rejected"


def test_invalid_transaction_conflicts_and_dangling():
    a = Node("n001", "cup")
    b = Node("n002", "table")
    graph = EditGraph((a, b), (Edge("e001", a.id, "on", b.id),))
    deletion = Operation("delete_node", before=b)
    with pytest.raises(ValueError, match="Dangling"):
        validate(graph, EditGraph(), (deletion,))
    with pytest.raises(ValueError, match="Multiple writes"):
        apply_operations(
            graph, (Operation("replace_node", a, Node(a.id, "bowl")), Operation("replace_node", a, Node(a.id, "plate")))
        )
    assert graph.nodes == (a, b)


def test_leakage_union_includes_source_caption_paths_and_target_pixels():
    rows = [row(identifier=f"r{i}", source=f"caption {i}") for i in range(5)]
    rows[1]["original_prompt"] = rows[0]["original_prompt"].upper()
    rows[2]["upstream_image_paths"] = rows[1]["upstream_image_paths"]
    images = {"r2": {"target_image": {"pixel_sha256": "same"}}, "r3": {"source_image": {"pixel_sha256": "same"}}}
    result, splits = runner.split_records(rows, ["r0"], 42, images)
    assert len({r["source_group_id"] for r in result[:4]}) == 1
    assert all(r["split"] == "train" for r in result[:4])
    reverse, other = runner.split_records(list(reversed(rows)), ["r0"], 42, images)
    assert splits == other
    assert {r["id"]: r["source_group_id"] for r in result} == {r["id"]: r["source_group_id"] for r in reverse}


class Response:
    def __init__(self, payload, start=0, size=None, status=206):
        self.status_code = status
        self.ok = status < 400
        self.content = payload
        self.raw = self
        self._data = io.BytesIO(payload)
        self.headers = {"Content-Range": f"bytes {start}-{start+len(payload)-1}/{size or len(payload)}"}
        self.text = payload.decode(errors="replace")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def raise_for_status(self):
        if not self.ok:
            raise ValueError("fixture_http_error")

    def read(self, n, **kwargs):
        return self._data.read(n)

    def json(self):
        return json.loads(self.content)


class RangeSession:
    def __init__(self, payload, status=206):
        self.payload, self.status, self.calls = payload, status, []

    def get(self, url, *, headers, **kwargs):
        start, end = map(int, headers["Range"][6:].split("-"))
        self.calls.append((start, end))
        return Response(self.payload[start : end + 1], start, len(self.payload), self.status)


def test_actual_parquet_projection_range_cache_and_corruption(tmp_path):
    blob = b"IMAGE_BYTES_MUST_NOT_BE_READ" * 20000
    table = pa.Table.from_pylist(
        [
            {
                "original_prompt": "A cat on a sofa.",
                "edited_prompt": "A dog on a sofa.",
                "edit_prompt": "Replace the cat with a dog.",
                "original_image": {"bytes": blob, "path": "source.png"},
                "edited_image": {"bytes": blob, "path": "target.png"},
            }
        ]
    )
    sink = io.BytesIO()
    pq.write_table(table, sink, compression="NONE")
    payload = sink.getvalue()
    session = RangeSession(payload)
    file = dataset.RangeFile("https://fixture/pinned.parquet", len(payload), tmp_path, session)
    p = pq.ParquetFile(file)
    result = p.read(columns=list(dataset.CAPTIONS)).to_pylist()
    assert result[0]["original_prompt"] == "A cat on a sofa."
    assert file.network_bytes < len(payload) / 3
    previous = len(session.calls)
    again = dataset.RangeFile("https://fixture/pinned.parquet", len(payload), tmp_path, session)
    assert pq.ParquetFile(again).read(columns=list(dataset.CAPTIONS)).to_pylist() == result
    assert len(session.calls) == previous
    first = next(tmp_path.glob("*.bin"))
    first.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="cache_corrupt"):
        pq.ParquetFile(dataset.RangeFile("https://fixture/pinned.parquet", len(payload), tmp_path, session)).read(
            columns=list(dataset.CAPTIONS)
        )


def test_range_server_ignoring_header_and_large_projection_rejected(tmp_path):
    file = dataset.RangeFile("https://fixture", 100, tmp_path, RangeSession(b"a" * 100, status=200))
    with pytest.raises(ValueError, match="honor_exact_range"):
        file.read(10)
    with pytest.raises(ValueError, match="metadata_budget"):
        dataset.RangeFile("https://fixture", 20_000_000, tmp_path).read(10_000_000)


@pytest.fixture
def offline_output(tmp_path, monkeypatch):
    output = tmp_path / "experiment"
    records = []
    for i in range(104):
        source, target, instruction, _ = SUPPORTED[i % len(SUPPORTED)]
        records.append(
            {
                "original_prompt": source,
                "edited_prompt": target,
                "edit_prompt": instruction,
                "original_image": {"path": f"source-{i}.png"},
                "edited_image": {"path": f"target-{i}.png"},
            }
        )
    parquet_path = tmp_path / "fixture.parquet"
    pq.write_table(pa.Table.from_pylist(records), parquet_path)
    p = pq.ParquetFile(parquet_path)
    source = {
        "repo": dataset.REPO,
        "revision": dataset.REVISION,
        "shards": [{"path": "data/fixture.parquet", "size": parquet_path.stat().st_size}],
        "splits": [{"name": "train", "num_examples": len(records)}],
        "physical_schema": str(p.schema_arrow),
    }
    write_once(output / "source.json", source)
    monkeypatch.setattr(dataset, "parquet", lambda *args: (pq.ParquetFile(parquet_path), None))
    dataset.plan(output, count=100)
    runner.prepare(output)
    monkeypatch.setattr(review, "ROOT", tmp_path)
    return output


def test_end_to_end_resume_review_and_predictor_contract(offline_output, monkeypatch):
    result = runner.compile_dataset(offline_output)
    path = result["compilation"]
    original = (offline_output / "compilations" / path.split("/")[-1] / "edits.json").read_bytes()
    monkeypatch.setattr(runner, "compile_record", lambda _: pytest.fail("resume reparsed a frozen label"))
    assert runner.compile_dataset(offline_output) == result
    assert (offline_output / "compilations" / path.split("/")[-1] / "edits.json").read_bytes() == original
    examples = runner.load_examples(path)
    assert len(examples["train"]) == 100
    example = examples["train"][0]
    assert set(example.model_input()) == {"source", "instruction"}
    assert set(feature_texts(**example.model_input())) == {"instruction", "nodes", "edges"}
    built = review.build(offline_output)
    packet = read(f"{built['directory']}/review-data.json")
    assert len(packet["diagnostic_review_ids"]) == 20
    assert len(packet["conditional_review_ids"]) == 20
    assert not any(r["reviewable"] for r in packet["cases"])
    assert not review.audit(offline_output)["ready"]
    with pytest.raises(ValueError, match="fewer_than_300"):
        runner.export(
            offline_output,
            None,
            f"{built['directory']}/review-template.csv",
            built["directory"],
            offline_output / "export",
        )


def test_immutable_plan_and_source_identity(offline_output):
    with pytest.raises(ValueError, match="immutable_evidence_changed"):
        dataset.plan(offline_output, count=101)
    path = offline_output / "metadata.json"
    data = read(path)
    data["records"][0]["original_prompt"] = "A cat."
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="metadata_evidence_changed"):
        runner.compile_dataset(offline_output)


def test_test_partition_never_passes_to_compiler(offline_output, monkeypatch):
    expanded = offline_output.parent / "expanded-fixture"
    write_once(expanded / "source.json", read(offline_output / "source.json"))
    dataset.plan(expanded, count=104)
    runner.prepare(expanded)
    original = runner.split_records

    def with_holdout(*args, **kwargs):
        rows, splits = original(*args, **kwargs)
        rows[-1]["split"] = "test"
        splits["train"].remove(rows[-1]["id"])
        splits["test"].append(rows[-1]["id"])
        return rows, splits

    monkeypatch.setattr(runner, "split_records", with_holdout)
    calls = []
    original_compile = runner.compile_record
    monkeypatch.setattr(runner, "compile_record", lambda r: (calls.append(r["id"]), original_compile(r))[1])
    result = runner.compile_dataset(expanded)
    assert len(calls) == 103
    assert result["summary"]["held_out_test_records_not_parsed"] == 1


def test_human_review_uncertainty_denominators_and_no_autoapproval():
    ids = [str(i) for i in range(20)]
    packet = {
        "cases": [{"id": i, "graphs_ready": True, "reviewable": True, "label": {"status": "candidate"}} for i in ids],
        "diagnostic_review_ids": ids,
        "conditional_review_ids": ids,
        "packet_sha256": "fixture",
    }
    decisions = {
        i: {
            "decision": "correct",
            "reviewer": "CPU fixture reviewer, not a real submission",
            "notes": "test only",
            **dict.fromkeys(AXES, "correct"),
        }
        for i in ids
    }
    assert (
        review.reliability(packet, {})["diagnostic_sample_wide"]["correct_fraction_entire_preselected_cohort"] is None
    )
    assert review.reliability(packet, decisions)["ready"]
    decisions["0"].update(decision="uncertain", operations="uncertain")
    report = review.reliability(packet, decisions)
    assert report["diagnostic_sample_wide"]["correct_fraction_entire_preselected_cohort"] == 19 / 20
    assert report["accepted_candidate_conditional"]["correct_fraction_among_determinate"] == 1
    assert not report["ready"]
    packet["cases"][1]["label"]["status"] = "rejected"
    with pytest.raises(ValueError, match="failed_candidate"):
        review.reliability(packet, decisions)


def test_review_packet_and_csv_tampering_rejected(offline_output):
    runner.compile_dataset(offline_output)
    built = review.build(offline_output)
    directory = built["directory"]
    template = read(f"{directory}/review-data.json")
    template["cases"][0]["instruction"] = "Fabricated instruction"
    template["cases"][0]["evidence_sha256"] = review.evidence(template["cases"][0])
    template["packet_sha256"] = digest({k: v for k, v in template.items() if k != "packet_sha256"})
    with open(f"{directory}/review-data.json", "w") as stream:
        json.dump(template, stream)
    with pytest.raises(ValueError, match="review_instruction_changed"):
        review.audit(offline_output)


def test_csv_evidence_tampering_rejected(offline_output):
    runner.compile_dataset(offline_output)
    built = review.build(offline_output)
    path = offline_output / "tampered-review.csv"
    packet = read(f"{built['directory']}/review-data.json")
    rows = list(csv.DictReader(io.StringIO(review.csv_bytes(packet, {}).decode("utf-8-sig"))))
    rows[0]["source_sha256"] = "tampered"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="review_csv_evidence_mismatch"):
        review.audit(offline_output, csv_path=path)


def test_loader_detects_leakage_after_manifest_rehash(offline_output):
    result = runner.compile_dataset(offline_output)
    from pathlib import Path

    path = Path(result["compilation"])
    manifest, edits = read(path / "manifest.json"), read(path / "edits.json")
    selected = manifest["records"][0]
    selected["split"] = "val"
    manifest["splits"]["train"].remove(selected["id"])
    manifest["splits"]["val"].append(selected["id"])
    for label in edits["records"]:
        if label["id"] == selected["id"]:
            label["split"] = "val"
    (path / "manifest.json").write_text(json.dumps(manifest))
    edits["dataset_manifest_sha256"] = sha256(path / "manifest.json")
    (path / "edits.json").write_text(json.dumps(edits))
    with pytest.raises(ValueError, match="source_family_leakage"):
        runner.load_examples(path)


def test_image_unavailability_preserved_and_resumed(tmp_path, monkeypatch):
    record = {**row(), "global_row": 123}
    calls = []

    def fail(*args, **kwargs):
        calls.append(args)
        return Response(b'{"error":"Scan size limit exceeded"}', status=500)

    monkeypatch.setattr(dataset.requests, "get", fail)
    result = dataset.prepare_images(tmp_path, [record])
    assert result["fixture"]["status"] == "unavailable"
    assert "viewer_http_500" in result["fixture"]["error"]
    assert dataset.prepare_images(tmp_path, [record]) == result
    assert len(calls) == 1


def test_image_adapter_downloads_only_matching_selected_images(tmp_path, monkeypatch):
    import tools.prepare_magicbrush as old_adapter

    monkeypatch.setattr(old_adapter, "ROOT", tmp_path)
    monkeypatch.setattr(dataset, "ROOT", tmp_path)
    record = {**row(), "global_row": 123}
    sink = io.BytesIO()
    Image.new("RGB", (8, 8), "blue").save(sink, format="PNG")
    image = sink.getvalue()
    remote = {
        **{k: record[k] for k in dataset.CAPTIONS},
        "original_image": {"src": dataset.SERVER + "/source.png"},
        "edited_image": {"src": dataset.SERVER + "/target.png"},
    }
    calls = []

    def get(url, **kwargs):
        calls.append(url)
        if url.endswith("/rows"):
            return Response(json.dumps({"rows": [{"row_idx": 123, "row": remote}]}).encode(), status=200)
        return Response(image, status=200)

    monkeypatch.setattr(dataset.requests, "get", get)
    result = dataset.prepare_images(tmp_path / "cache", [record])
    assert result["fixture"]["status"] == "available"
    assert len(calls) == 3
    assert result["fixture"]["source_image"]["sha256"] == hashlib.sha256(image).hexdigest()
    assert dataset.prepare_images(tmp_path / "cache", [record]) == result
    remote["edited_prompt"] = "A different revision"
    other = dataset.prepare_images(tmp_path / "other", [record])
    assert "revision_or_caption_mismatch" in other["fixture"]["error"]
    assert len(calls) == 4


def test_explicit_image_retry_keeps_original_failure_and_resolves(tmp_path, monkeypatch):
    import tools.prepare_magicbrush as old_adapter

    monkeypatch.setattr(old_adapter, "ROOT", tmp_path)
    monkeypatch.setattr(dataset, "ROOT", tmp_path)
    record = {**row(), "global_row": 123}
    monkeypatch.setattr(dataset.requests, "get", lambda *a, **k: Response(b"rate limited", status=429))
    first = dataset.prepare_images(tmp_path / "cache", [record])
    assert first["fixture"]["status"] == "unavailable"
    sink = io.BytesIO()
    Image.new("RGB", (4, 4), "red").save(sink, format="PNG")
    remote = {
        **{k: record[k] for k in dataset.CAPTIONS},
        "original_image": {"src": dataset.SERVER + "/s.png"},
        "edited_image": {"src": dataset.SERVER + "/t.png"},
    }

    def get(url, **kwargs):
        if url.endswith("/rows"):
            return Response(json.dumps({"rows": [{"row_idx": 123, "row": remote}]}).encode(), status=200)
        return Response(sink.getvalue(), status=200)

    monkeypatch.setattr(dataset.requests, "get", get)
    resolved = dataset.prepare_images(tmp_path / "cache", [record], retry=True)
    assert resolved["fixture"]["status"] == "available"
    assert read(tmp_path / "cache/image-receipts" / f"{digest('fixture')}.json") == first["fixture"]
    assert runner.image_receipts(tmp_path / "cache") == resolved
    assert dataset.prepare_images(tmp_path / "cache", [record]) == resolved


def test_rate_limit_stops_further_image_network_requests(tmp_path, monkeypatch):
    calls = []

    def limited(*args, **kwargs):
        calls.append(args)
        return Response(b"rate limited", status=429)

    monkeypatch.setattr(dataset.requests, "get", limited)
    rows = [{**row(identifier=str(i)), "global_row": i} for i in range(3)]
    result = dataset.prepare_images(tmp_path, rows)
    assert len(calls) == 1
    assert all(r["status"] == "unavailable" for r in result.values())
    assert "deferred_after_rate_limit" in result["2"]["error"]


def test_predictor_cpu_collation_uses_only_source_and_instruction(offline_output):
    import torch

    from tools.edit_model_v1.model import collate_inputs, collate_targets

    result = runner.compile_dataset(offline_output)
    examples = runner.load_examples(result["compilation"])["train"][:3]

    class FixtureEmbeddings:
        dim = 16

        def __getitem__(self, text):
            return torch.tensor([hashlib.sha256(text.encode()).digest()[i] / 255 for i in range(self.dim)])

    batch = collate_inputs([e.model_input() for e in examples], FixtureEmbeddings(), device="cpu")
    assert batch["instruction"].shape == (3, 16)
    assert batch["nodes"].shape == (3, 20, 16)
    before, after = collate_targets(examples, device="cpu")
    assert before.shape == after.shape
    assert not any(k in batch for k in ("target", "target_graph", "edited_prompt", "edited_image", "operations"))


def test_reports_include_failed_diagnostic_denominator():
    accepted = compile_record(row())
    failed = compile_record(row(source="An oil painting of a car.", identifier="failed"))
    report = summarize([accepted, failed])
    assert report["acceptance_rate"] == 0.5
    assert report["stages"]["caption_parsing"]["diagnostic_denominator"] == 2
    assert report["operation_distribution_candidates"]["replace_node"] == 1
    assert report["visual_accuracy"] is None
