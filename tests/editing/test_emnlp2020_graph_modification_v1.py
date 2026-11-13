"""CPU tests. Synthetic fixtures are never pilot/300-example evidence."""

import copy
import csv
import io
import json
import os
import pickle
import pickletools

import networkx as nx
import pytest

from src.editing.operations import Edge, EditGraph, Node, Operation, apply_operations
from tools.edit_model_v1.codec import canonical_source, encode_actions
from tools.edit_model_v1.data import feature_texts
from tools.emnlp2020_graph_modification_v1.alignment import AlignmentError, align, attribute_policy, canonical_raw
from tools.emnlp2020_graph_modification_v1.compiler import compile_record
from tools.emnlp2020_graph_modification_v1.data import checked_inputs, leakage_filter, load_examples, source_fingerprint
from tools.emnlp2020_graph_modification_v1.dataset import (
    ARCHIVE,
    ARCHIVE_SHA256,
    code_identity,
    digest,
    fixed,
    inspect,
    member_bytes,
    read,
    sandbox_scan,
    select_indices,
    sha256,
    verified_archive,
    write,
)
from tools.emnlp2020_graph_modification_v1.report import summarize
from tools.emnlp2020_graph_modification_v1.review import AXES, FIELDS, audit, build, checked_packet, reliability
from tools.emnlp2020_graph_modification_v1.runner import compile_dataset, expansion, export
from tools.emnlp2020_graph_modification_v1.safe_pickle import parse_record, scan


def raw(labels, edges=(), directed=True):
    return {
        "directed": directed,
        "nodes": [{"id": i, "data": {"feature": label}} for i, label in labels.items()],
        "edges": [{"u": u, "v": v, "data": {"feature": p}} for u, p, v in edges],
        "graph_metadata": {},
    }


def record(source, target, index=0, split="train"):
    return {
        "id": f"fixture_{index:06d}",
        "source_dataset": "synthetic_unit_fixture",
        "source_variant": "fixture",
        "official_split": "train" if split == "train" else "dev",
        "split": split,
        "original_record_index": index,
        "instruction": f"Change fixture {index}.",
        "raw_source": source,
        "raw_target": target,
        "load_errors": [],
    }


@pytest.mark.parametrize("protocol", [3, 4, 5])
@pytest.mark.parametrize("directed", [False, True])
def test_nonexecuting_networkx_state_reader(protocol, directed):
    graph = nx.DiGraph() if directed else nx.Graph()
    graph.add_node(5, feature="cat")
    graph.add_node(2, feature="black")
    graph.add_edge(5, 2, feature="att")
    _ = graph.nodes  # Include cached NodeView as in the real release.
    payload = pickle.dumps(graph, protocol=protocol)
    result = parse_record(pickletools.genops(payload))
    assert result["directed"] == directed
    assert {r["data"]["feature"] for r in result["nodes"]} == {"cat", "black"}
    assert len(result["edges"]) == 1


def test_malicious_pickle_never_executes(tmp_path):
    sentinel = tmp_path / "executed"

    class Attack:
        def __reduce__(self):
            return os.system, (f"touch {sentinel}",)

    result = scan(pickle.dumps(Attack(), protocol=3), [0])
    assert "forbidden_global" in result["selected"]["0"]["error"]
    assert not sentinel.exists()


def test_reduce_rejected_even_for_allowed_global():
    with pytest.raises(ValueError, match="forbidden_opcode:REDUCE"):
        parse_record(pickletools.genops(b"\x80\x03cbuiltins\ndict\n)R."))


def test_unknown_metadata_not_dropped():
    graph = nx.Graph(description="do not drop")
    with pytest.raises(ValueError, match="nonempty_graph_metadata"):
        parse_record(pickletools.genops(pickle.dumps(graph, protocol=3)))


def test_truncated_pickle_is_not_silent_eof():
    with pytest.raises(ValueError):
        scan(b"\x80\x03}", [0])


@pytest.mark.skipif(os.geteuid() != 0, reason="requires chroot capability, never falls back to unconfined unpickling")
def test_sandbox_stream_selection_and_limits():
    graph = nx.Graph()
    graph.add_node(0, feature="dog")
    payload = pickle.dumps(graph, protocol=3)
    result = sandbox_scan(payload * 3, [1])
    assert result["record_count"] == 3 and set(result["selected"]) == {"1"}
    assert "chroot" in result["isolation"] and "no_open_socket_exec" in result["isolation"]


@pytest.mark.parametrize(
    "source,target,expected",
    [
        (raw({0: "dog"}), raw({9: "dog", 2: "ball"}, [(9, "near", 2)]), {"add_node": 1, "add_edge": 1}),
        (raw({0: "dog", 1: "ball"}, [(0, "near", 1)]), raw({9: "dog"}), {"delete_node": 1, "delete_edge": 1}),
        (raw({0: "dog"}), raw({0: "cat"}), {"replace_node": 1}),
        (raw({0: "dog", 1: "ball"}), raw({7: "dog", 8: "ball"}, [(7, "near", 8)]), {"add_edge": 1}),
        (raw({0: "dog", 1: "ball"}, [(0, "near", 1)]), raw({7: "dog", 8: "ball"}), {"delete_edge": 1}),
        (
            raw({0: "dog", 1: "ball"}, [(0, "near", 1)]),
            raw({7: "dog", 8: "ball"}, [(7, "under", 8)]),
            {"replace_edge": 1},
        ),
    ],
)
def test_six_operation_types_and_reconstruction(source, target, expected):
    result = compile_record(record(source, target))
    assert result["status"] == "candidate", result["problems"]
    assert {k: v for k, v in result["operation_support"].items() if v} == expected
    left, right = (EditGraph.from_dict(result[k]) for k in ("source_graph", "target_graph"))
    ops = tuple(Operation.from_dict(o) for o in result["operations"])
    assert apply_operations(left, ops) == right
    assert len(encode_actions(ops)) - 1 <= 2048


def test_attribute_topology_and_refused_flattening():
    left = raw({0: "car", 1: "red"}, [(0, "att", 1)], False)
    right = raw({5: "car", 8: "blue"}, [(5, "att", 8)], False)
    result = compile_record(record(left, right))
    assert result["status"] == "candidate"
    assert result["operation_support"]["replace_node"] == 1
    assert len(result["source_graph"]["nodes"]) == 2 and len(result["source_graph"]["edges"]) == 2
    assert not result["lossy_conversion"]
    with pytest.raises(ValueError, match="ownership_unproven"):
        attribute_policy(left, "B")
    assert attribute_policy(raw({0: "dog"}), "B")[1]["lossy"] is False


def test_attribute_subtree_deletion_and_reordered_ids():
    left = raw(
        {0: "man", 1: "pants", 2: "black", 3: "hat"}, [(0, "wearing", 1), (1, "att", 2), (0, "wearing", 3)], False
    )
    right = raw({0: "man", 1: "hat"}, [(0, "wearing", 1)], False)
    result = compile_record(record(left, right))
    assert result["status"] == "candidate"
    assert result["operation_support"]["delete_node"] == 2
    assert result["operation_support"]["delete_edge"] == 4
    assert any(r["source_raw_id"] == 3 and r["target_raw_id"] == 1 for r in result["alignment"]["node_correspondence"])


def test_identical_nodes_and_symmetry_are_ambiguous():
    with pytest.raises(AlignmentError, match="ambiguous_node_identity"):
        align(raw({0: "dog", 1: "dog"}), raw({0: "dog"}))


def test_identical_labels_disambiguated_only_by_structure():
    left = raw({0: "dog", 1: "dog", 2: "cat"}, [(0, "near", 2)])
    right = raw({8: "dog", 9: "dog", 7: "cat"}, [(9, "near", 7)])
    _, _, evidence = align(left, right)
    assert {r["target_raw_id"]: r["source_raw_id"] for r in evidence["node_correspondence"]} == {9: 0, 8: 1, 7: 2}


def test_edge_endpoint_change_not_fabricated_replacement():
    left = raw({0: "dog", 1: "cat", 2: "ball"}, [(0, "near", 2)])
    right = raw({0: "dog", 1: "cat", 2: "ball"}, [(1, "near", 2)])
    result = compile_record(record(left, right))
    assert result["status"] == "candidate"
    assert result["operation_support"]["replace_edge"] == 0
    assert result["operation_support"]["delete_edge"] == result["operation_support"]["add_edge"] == 1


def test_directed_relation_and_undirected_reciprocal_semantics():
    directed = raw({0: "dog", 1: "cat"}, [(0, "near", 1)])
    undirected = {**directed, "directed": False}
    dg, _ = canonical_raw(directed)
    ug, _ = canonical_raw(undirected)
    assert len(dg.edges) == 1 and len(ug.edges) == 2
    assert {(e.subject, e.object) for e in ug.edges} == {("n001", "n002"), ("n002", "n001")}
    with pytest.raises(AlignmentError, match="directedness_changed"):
        align(directed, undirected)


def test_noop_and_missing_values_have_explicit_denominators():
    noop = compile_record(record(raw({0: "dog"}), raw({9: "dog"})))
    assert noop["no_graph_changes"] and noop["status"] == "rejected"
    broken = compile_record({**record(None, None), "load_errors": ["missing"]})
    report = summarize([broken], [broken])
    assert report["stages"]["graph_diff"]["not_evaluated"] == 1
    assert report["no_graph_changes"] is None and report["lossy_graph_conversions"] is None
    assert all(v is None for v in report["operation_distribution_all_derived"].values())


@pytest.mark.parametrize(
    "edges", [[(0, "near", 4)], [(0, "near", 1), (0, "near", 1)], [(0, "near", 1), (0, "under", 1)]]
)
def test_invalid_raw_edges_not_silently_deduplicated(edges):
    result = compile_record(record(raw({0: "dog", 1: "cat"}, edges), raw({0: "dog"})))
    assert result["status"] == "rejected" and not result["stages"]["graph_parsing"]


def test_invalid_transactions_remain_strict():
    source = EditGraph((Node("a", "dog"), Node("b", "cat")), (Edge("x", "a", "near", "b"),))
    with pytest.raises(ValueError, match="Dangling"):
        apply_operations(source, [Operation("delete_node", before=source.nodes[0])])
    op = Operation("replace_node", before=source.nodes[0], after=Node("a", "horse"))
    with pytest.raises(ValueError, match="Multiple writes"):
        apply_operations(source, [op, op])


def test_target_independent_source_ids_and_features():
    source = raw({3: "dog", 9: "cat"}, [(3, "near", 9)])
    first = compile_record(record(source, raw({0: "dog", 1: "horse"}, [(0, "near", 1)])))
    second = compile_record(record(source, raw({0: "dog", 1: "cow"}, [(0, "near", 1)])))
    expected = canonical_source(canonical_raw(source)[0])[0]
    assert first["source_graph"] == second["source_graph"] == expected.to_dict()
    assert feature_texts(expected, "command")["nodes"] == ['["dog", []]', '["cat", []]']


def test_fingerprint_reordering_and_duplicate_split_leakage():
    a = raw({0: "dog", 1: "cat"}, [(0, "near", 1)])
    b = raw({5: "cat", 9: "dog"}, [(9, "near", 5)])
    assert source_fingerprint(a) == source_fingerprint(b)
    rows = [record(a, a, 0), record(b, b, 1, "val"), {**record(b, b, 2), "instruction": "Change fixture 0."}]
    exclusions = leakage_filter(rows)
    assert exclusions == {
        "fixture_000001": "duplicate_source_graph_across_official_splits",
        "fixture_000002": "duplicate_source_instruction_pair",
    }


def test_immutable_writes_and_checksum_checks(tmp_path):
    p = tmp_path / "artifact"
    fixed(p, b"abc")
    fixed(p, b"abc")
    with pytest.raises(ValueError, match="immutable_output_changed"):
        fixed(p, b"different")
    with pytest.raises(ValueError, match="checksum_mismatch"):
        verified_archive(p)
    assert select_indices(30051) == select_indices(30051) and len(select_indices(30051)) == 100
    with pytest.raises(ValueError, match="not_enough"):
        select_indices(99)


def synthetic_artifacts(output, count=25):
    # Build loader artifacts directly: these are test fixtures, never a real pilot.
    originals = [
        record(raw({0: f"dog{i}"}), raw({1: f"cat{i}"}), i, "val" if i == count - 1 else "train") for i in range(count)
    ]
    records = [compile_record(r) for r in originals]
    assert not leakage_filter(records)
    write(output / "raw-records.json", originals)
    write(output / "cohort.json", {"fixture": True})
    write(output / "raw-receipt.json", {"fixture": True})
    manifest = {
        "archive_sha256": ARCHIVE_SHA256,
        "records": [
            {
                k: v
                for k, v in r.items()
                if k
                in (
                    "id",
                    "split",
                    "official_split",
                    "instruction",
                    "original_record_index",
                    "source_group_id",
                    "source_instruction_sha256",
                )
            }
            | {"raw_evidence_sha256": digest(original)}
            for r, original in zip(records, originals, strict=True)
        ],
        "splits": {s: [r["id"] for r in records if r["split"] == s] for s in ("train", "val", "test")},
        "review_ids": [r["id"] for r in records[:20]],
        "inputs": {"raw-records.json": sha256(output / "raw-records.json")},
        "code": code_identity(),
    }
    write(output / "manifest.json", manifest)
    write(output / "edits.json", {"dataset_manifest_sha256": sha256(output / "manifest.json"), "records": records})
    return output


@pytest.fixture
def synthetic_compilation(tmp_path):
    return synthetic_artifacts(tmp_path / "synthetic")


def test_existing_example_loader_contract(synthetic_compilation):
    examples, report = load_examples(synthetic_compilation)
    assert report["examples"] == {"train": 24, "val": 1}
    e = examples["train"][0]
    assert set(e.model_input()) == {"source", "instruction"}
    assert apply_operations(e.source, e.operations) == e.target
    assert not report["test_labels_decoded"]


def test_loader_manifest_and_raw_tamper_detection(synthetic_compilation):
    path = synthetic_compilation / "raw-records.json"
    rows = read(path)
    rows[0]["instruction"] = "tampered"
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="input_evidence_changed"):
        checked_inputs(synthetic_compilation)


def test_loader_rejects_target_derived_source_or_fabricated_actions(synthetic_compilation):
    path = synthetic_compilation / "edits.json"
    payload = read(path)
    payload["records"][0]["source_graph"]["nodes"][0]["label"] = "oracle"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="normalized_label_differs_from_raw"):
        load_examples(synthetic_compilation)


def test_resume_does_not_recompute_changed_evidence(synthetic_compilation):
    output = synthetic_compilation
    write(output / "summary.json", {"fixture": True})
    write(output / "output-receipt.json", {"files": {"edits.json": sha256(output / "edits.json")}})
    receipt = {
        "archive_sha256": ARCHIVE_SHA256,
        "cohort_sha256": sha256(output / "cohort.json"),
        "raw_sha256": sha256(output / "raw-records.json"),
    }
    (output / "raw-receipt.json").write_text(json.dumps(receipt))
    assert compile_dataset(output) == {"fixture": True}
    (output / "edits.json").write_text("{}")
    with pytest.raises(ValueError, match="compilation_output_hash_changed"):
        compile_dataset(output)


def test_review_blank_pending_and_no_export(synthetic_compilation, tmp_path):
    directory = tmp_path / "review"
    build(synthetic_compilation, directory)
    packet = checked_packet(synthetic_compilation, directory)
    gate = audit(synthetic_compilation, directory)
    assert not gate["ready"] and gate["reviewed_pairs"] == 0
    assert gate["candidate_conditional"]["correct_fraction_among_determinate"] is None
    assert packet["human_reviews_created"] == 0 and len(packet["cases"]) >= 20
    template = (directory / "review-template.csv").read_text(encoding="utf-8-sig")
    assert next(csv.DictReader(io.StringIO(template)))["reviewer"] == ""
    with pytest.raises(ValueError, match="requires_real_human"):
        expansion(synthetic_compilation, ARCHIVE, tmp_path / "expanded", directory, None)
    with pytest.raises(ValueError, match="human_review_gate_not_ready"):
        export(
            synthetic_compilation,
            tmp_path / "export.json",
            synthetic_compilation,
            directory,
            directory / "review-template.csv",
        )


def test_review_axes_names_evidence_and_tampering(synthetic_compilation, tmp_path):
    directory = tmp_path / "review"
    build(synthetic_compilation, directory)
    packet = checked_packet(synthetic_compilation, directory)
    # Synthetic reviewer exists only in a synthetic unit test.
    reviews = {
        r["id"]: {"decision": "correct", "reviewer": "UNIT_TEST_ONLY", **{a: "correct" for a in AXES}}
        for r in packet["cases"]
    }
    assert reliability(packet, reviews)["ready"]
    first = packet["cases"][0]["id"]
    reviews[first]["reviewer"] = ""
    with pytest.raises(ValueError, match="reviewer"):
        reliability(packet, reviews)
    rows = list(csv.DictReader((directory / "review-template.csv").open(encoding="utf-8-sig")))
    rows[0]["evidence_sha256"] = "changed"
    target = tmp_path / "changed.csv"
    with target.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="evidence_mismatch"):
        audit(synthetic_compilation, directory, target)
    changed = copy.deepcopy(packet)
    changed["cases"][0]["label"]["instruction"] = "tampered"
    (directory / "review-data.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="evidence_or_selection_changed"):
        checked_packet(synthetic_compilation, directory)


def test_graph_search_and_operation_budgets():
    too_large = compile_record(record(raw({i: str(i) for i in range(8)}), raw({i: str(i) for i in range(8)})))
    assert "alignment_search_budget_exceeded" in too_large["problems"]
    labels = {i: f"item{i}" for i in range(7)}
    edges = [(i, "near", j) for i in labels for j in labels if i < j]
    too_many = compile_record(record(raw({0: "item0"}, directed=False), raw(labels, edges, False)))
    assert "operation_count_budget_exceeded" in too_many["problems"]
    assert too_many["operation_count"] > 12


def test_review_gate_export300_synthetic_contract_only(tmp_path):
    output = synthetic_artifacts(tmp_path / "synthetic300", 301)
    directory = tmp_path / "review"
    build(output, directory)
    packet = checked_packet(output, directory)
    reviews = {
        r["id"]: {"decision": "correct", "reviewer": "UNIT_TEST_ONLY", **{a: "correct" for a in AXES}}
        for r in packet["cases"]
    }
    from tools.emnlp2020_graph_modification_v1.review import csv_bytes

    csv_path = tmp_path / "synthetic-only.csv"
    csv_path.write_bytes(csv_bytes(packet, reviews))
    result = export(output, tmp_path / "synthetic-only-export.json", output, directory, csv_path)
    assert result["examples"] == {"train": 300, "val": 1}
    payload = read(tmp_path / "synthetic-only-export.json")
    assert not payload["assignment_dataset_source_requirement_satisfied"]
    # Changing an axis cannot manufacture an overall correct review.
    reviews[packet["cases"][0]["id"]]["alignment"] = "incorrect"
    with pytest.raises(ValueError, match="contradicts"):
        csv_bytes(packet, reviews)


@pytest.mark.skipif(os.geteuid() != 0, reason="root chroot required")
def test_sandbox_denies_files_network_and_process_execution(tmp_path):
    import subprocess
    import sys

    from tools.emnlp2020_graph_modification_v1 import safe_pickle

    jail = tmp_path / "jail"
    jail.mkdir()
    code = """
import os, socket, subprocess, sys, runpy
module=runpy.run_path(sys.argv[1])
module['confine'](sys.argv[2])
for action in (lambda: open('/etc/passwd'), lambda: socket.socket(), lambda: subprocess.run(['/bin/true'])):
    try:
        action()
    except OSError:
        continue
    raise RuntimeError('sandbox escape')
print('all_denied')
"""
    process = subprocess.run(
        [sys.executable, "-I", "-c", code, safe_pickle.__file__, str(jail)],
        env={"LC_ALL": "C"},
        cwd=jail,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert process.returncode == 0, process.stderr
    assert process.stdout.strip() == "all_denied"


def test_review_browser_graph_only_evidence_and_csv(synthetic_compilation, tmp_path):
    import threading
    from http.server import ThreadingHTTPServer

    from tools.emnlp2020_graph_modification_v1.review import handler

    playwright = pytest.importorskip("playwright.sync_api")
    directory = tmp_path / "review"
    build(synthetic_compilation, directory)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler(synthetic_compilation, directory))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        with playwright.sync_playwright() as runtime:
            browser = runtime.chromium.launch(args=["--disable-gpu"])
            page = browser.new_page()
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(f"http://127.0.0.1:{httpd.server_port}/")
            assert page.locator("main section").count() == 4
            assert page.locator("img").count() == 0
            page.locator("#reviewer").fill("UNIT_TEST_ONLY")
            for axis in AXES:
                page.locator("#" + axis).select_option("correct")
            page.locator("#decision").select_option("correct")
            page.locator("#save").click()
            page.wait_for_function("document.getElementById('saved').textContent.includes('ready=false')")
            files = list((directory / "submissions").glob("*.csv"))
            assert len(files) == 1
            assert audit(synthetic_compilation, directory, files[0])["reviewed_pairs"] == 1
            page.locator("#next").click()
            page.locator("#prev").click()
            assert page.locator("#decision").input_value() == "correct"
            assert not errors
            browser.close()
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()


@pytest.mark.skipif(
    not ARCHIVE.exists() or os.geteuid() != 0, reason="official local download and constrained reader required"
)
def test_real_official_crowdsourced100_pinned_cohort():
    inspection = inspect(ARCHIVE)
    assert inspection["train_instruction_count"] == 30051
    assert inspection["archive_sha256"] == ARCHIVE_SHA256
    indices = select_indices(30051)
    source = sandbox_scan(member_bytes(ARCHIVE, "train", "src_graph.bin"), indices)
    target = sandbox_scan(member_bytes(ARCHIVE, "train", "tgt_graph.bin"), indices)
    assert source["record_count"] == target["record_count"] == 30051
    assert len(source["selected"]) == 100
    instructions = member_bytes(ARCHIVE, "train", "src_text.txt").decode().splitlines()
    for i in indices:
        left, right = (r["selected"][str(i)] for r in (source, target))
        assert left["error"] is right["error"] is None
        row = {
            **record(left["graph"], right["graph"], i),
            "instruction": instructions[i],
            "source_dataset": "SceneGraphModification",
        }
        result = compile_record(row)
        assert result["stages"]["transaction_validity"], result["problems"]
        assert result["raw_source"]["directed"] is False
    with pytest.raises(ValueError, match="never_test"):
        member_bytes(ARCHIVE, "test", "src_graph.bin")
