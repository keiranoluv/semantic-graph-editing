"""CPU-only synthetic fixtures. No real reviewer decisions or GPU inference."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from src.editing.operations import EditGraph, Operation, apply_operations
from tools.edit_model_v1.data import load_examples
from tools.magicbrush_joint_v1 import client, prepare, review, runner
from tools.magicbrush_joint_v1.compiler import compile_annotation
from tools.magicbrush_joint_v1.schema import SCHEMA, VERSION, parse
from tools.magicbrush_joint_v1.storage import read, sample_dir, write_once
from tools.review_magicbrush import AXES, export_reviews

LIMITS = {"max_nodes": 20, "max_edges": 60, "max_operations": 12}


def node(identifier, label, attrs=()):
    return {
        "id": identifier,
        "label": label,
        "attributes": list(attrs),
        "evidence": "Visible at the center of this image",
    }


def edge(identifier, subject, predicate, obj):
    return {
        "id": identifier,
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "evidence": "Visible spatial relation in this image",
    }


def link(source, target):
    return {
        "source_id": source,
        "target_id": target,
        "certainty": "clear",
        "evidence": "Same visual location and scene role in both images",
    }


def change(kind, source, target, action):
    return {
        "entity_type": kind,
        "source_id": source,
        "target_id": target,
        "change": action,
        "source_evidence": "Source region shows the old state or clear absence",
        "target_evidence": "Same target region shows the new state or clear absence",
        "unintended": False,
    }


def annotation():
    return {
        "schema_version": VERSION,
        "instruction_fulfilled": "yes",
        "annotation_status": "supported",
        "source_entities": [node("s1", "dog"), node("s2", "tree")],
        "target_entities": [node("t9", "cat"), node("t8", "tree")],
        "source_relations": [edge("r1", "s1", "next_to", "s2")],
        "target_relations": [edge("r9", "t9", "next_to", "t8")],
        "entity_correspondences": [link("s1", "t9"), link("s2", "t8")],
        "relation_correspondences": [link("r1", "r9")],
        "observed_changes": [change("node", "s1", "t9", "replacement")],
        "unchanged_entities": ["s2"],
        "uncertainties": [],
        "explanation": "Dog replaced by cat, with the tree and relation preserved",
        "model_confidence": 0.95,
    }


def check(a):
    result = compile_annotation(a, LIMITS)
    source, target = (EditGraph.from_dict(result[k]) for k in ("source_graph", "target_graph"))
    ops = [Operation.from_dict(r) for r in result["operations"]]
    assert apply_operations(source, ops) == target
    assert result["human_review_status"] == "pending"
    return result, [r.kind for r in ops]


def test_schema_and_object_replacement():
    a = annotation()
    assert parse(json.dumps(a)) == a
    result, kinds = check(a)
    assert kinds == ["replace_node"]
    assert result["classification"] == "candidate_accepted"
    assert result["source_graph"]["nodes"][0]["id"] == result["target_graph"]["nodes"][0]["id"]


@pytest.mark.parametrize("raw", ["{bad", '{"schema_version":"x"}', '{"a":1,"a":2}', "```json\n{}\n```"])
def test_malformed_json_or_schema(raw):
    with pytest.raises(ValueError):
        parse(raw)


def test_schema_matches_jsonschema_when_available():
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.Draft202012Validator.check_schema(SCHEMA)
    jsonschema.validate(annotation(), SCHEMA)


@pytest.mark.parametrize(
    "mutation,reason",
    [
        (lambda a: a["entity_correspondences"].pop(), "missing_correspondence"),
        (lambda a: a["entity_correspondences"][0].update(certainty="uncertain"), "ambiguous_correspondence"),
        (lambda a: a["entity_correspondences"].append(link("s1", "t8")), "multiple_correspondence"),
        (lambda a: a.update(observed_changes=[]), "observed_changes_graph_mismatch"),
        (lambda a: a.update(unchanged_entities=["s1", "s2"]), "unexpected_modification"),
        (lambda a: a["target_relations"][0].update(subject="unknown"), "dangling_relation_endpoint"),
        (lambda a: a["source_entities"][0].update(attributes=["red", "RED"]), "Repeated node attribute"),
    ],
)
def test_reject_inconsistent_observations(mutation, reason):
    a = annotation()
    mutation(a)
    with pytest.raises(ValueError, match=reason):
        check(a)


def test_add_node_and_relation():
    a = annotation()
    a["target_entities"].append(node("t3", "bird"))
    a["entity_correspondences"].append(link(None, "t3"))
    a["target_relations"].append(edge("r3", "t3", "on", "t8"))
    a["relation_correspondences"].append(link(None, "r3"))
    a["observed_changes"] += [change("node", None, "t3", "addition"), change("edge", None, "r3", "addition")]
    _, kinds = check(a)
    assert sorted(kinds) == ["add_edge", "add_node", "replace_node"]


def test_removal_with_explicit_incident_edge():
    a = annotation()
    a["target_entities"] = [node("t8", "tree")]
    a["entity_correspondences"][0] = link("s1", None)
    a["target_relations"] = []
    a["relation_correspondences"] = [link("r1", None)]
    a["observed_changes"] = [change("node", "s1", None, "removal"), change("edge", "r1", None, "removal")]
    _, kinds = check(a)
    assert sorted(kinds) == ["delete_edge", "delete_node"]


def test_attribute_replacement():
    a = annotation()
    a["source_entities"][0] = node("s1", "car", ["red"])
    a["target_entities"][0] = node("t9", "car", ["blue"])
    result, kinds = check(a)
    assert kinds == ["replace_node"]
    assert result["operations"][0]["before"]["label"] == result["operations"][0]["after"]["label"]


def test_relation_replacement():
    a = annotation()
    a["target_entities"][0] = node("t9", "dog")
    a["target_relations"][0]["predicate"] = "behind"
    a["unchanged_entities"] = ["s1", "s2"]
    a["observed_changes"] = [change("edge", "r1", "r9", "replacement")]
    _, kinds = check(a)
    assert kinds == ["replace_edge"]


def test_duplicate_edges_rejected():
    a = annotation()
    a["source_relations"].append(edge("r2", "s1", "next_to", "s2"))
    a["relation_correspondences"].append(link("r2", None))
    with pytest.raises(ValueError, match="Duplicate semantic edge"):
        check(a)


def test_invalid_and_conflicting_operations():
    result, _ = check(annotation())
    source = EditGraph.from_dict(result["source_graph"])
    op = Operation.from_dict(result["operations"][0])
    with pytest.raises(ValueError, match="Multiple writes"):
        apply_operations(source, [op, op])
    with pytest.raises(ValueError, match="requires after"):
        Operation.from_dict({"kind": "add_node", "before": None, "after": None})
    with pytest.raises(ValueError, match="Invalid operation JSON"):
        Operation.from_dict({"kind": "invent_object", "before": None, "after": None})


@pytest.mark.parametrize(
    "status,fulfilled", [("unsuccessful", "no"), ("ambiguous", "uncertain"), ("supported", "partial")]
)
def test_nonfulfilled_never_training_candidate(status, fulfilled):
    a = annotation()
    a.update(annotation_status=status, instruction_fulfilled=fulfilled)
    result, _ = check(a)
    assert result["classification"] == "needs_review"
    assert not result["eligible_for_training"]


def test_style_only_and_noop():
    a = annotation()
    a["target_entities"][0] = node("t9", "dog")
    a.update(annotation_status="unsupported", observed_changes=[], unchanged_entities=["s1", "s2"])
    result, kinds = check(a)
    assert kinds == [] and result["classification"] == "rejected"
    assert not result["eligible_for_training"]
    a.update(annotation_status="supported")
    assert check(a)[0]["classification"] == "needs_review"


def test_uncertainty_unintended_and_limits():
    a = annotation()
    a["observed_changes"][0]["unintended"] = True
    assert not check(a)[0]["eligible_for_training"]
    a["observed_changes"][0]["unintended"] = False
    a["uncertainties"] = ["Source image is blurry"]
    assert not check(a)[0]["eligible_for_training"]
    with pytest.raises(ValueError, match="graph_size_limit"):
        compile_annotation(a, {**LIMITS, "max_nodes": 1})
    with pytest.raises(ValueError, match="operation_count_limit"):
        compile_annotation(a, {**LIMITS, "max_operations": 0})


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    # All fixture images and invented review decisions stay inside pytest tmp_path.
    from tools import review_magicbrush

    for module in (prepare, client, runner, review, review_magicbrush):
        monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(prepare, "source_hashes", lambda: {"fixture": "cpu-only"})
    monkeypatch.setattr(prepare, "revision", lambda: "a" * 40)
    records = []
    for i in range(24):
        split = "train" if i < 22 else "val" if i == 22 else "test"
        row = {
            "id": f"pair:{i:03d}",
            "img_id": str(i),
            "turn_index": 1,
            "source_group_id": f"group:{i}",
            "split": split,
            "official_split": "train",
            "instruction": [
                "Add a bird",
                "Remove the dog",
                "Replace the dog",
                "Make it red",
                "Move it behind the tree",
                "Make a surreal painting",
            ][i % 6],
        }
        for offset, role in enumerate(("source_image", "target_image")):
            path = tmp_path / f"image-{i}-{offset}.png"
            image = Image.new("RGB", (4, 3), (i, offset, 90))
            image.save(path)
            row[role] = {
                "path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "pixel_sha256": hashlib.sha256(b"4:3:RGB:" + image.tobytes()).hexdigest(),
                "width": 4,
                "height": 3,
            }
        records.append(row)
    m = {
        "records": records,
        "splits": {s: [r["id"] for r in records if r["split"] == s] for s in ("train", "val", "test")},
        "revision": "fixture",
        "review_ids": [],
    }
    write_once(tmp_path / "manifest.json", m)
    cfg = read(prepare.DEFAULT_CONFIG)
    cfg.update(
        manifest="manifest.json",
        output="data/processed/level2/magicbrush-joint-fixture-v1",
        reviews="reviews/magicbrush-joint-fixture-v1",
    )
    write_once(tmp_path / "config.json", cfg)
    cfg = prepare.config(tmp_path / "config.json")
    prepare.prepare(cfg)
    return cfg, m, tmp_path / cfg["output"]


class FakeClient:
    def __init__(self, responses=None):
        self.calls = 0
        self.responses = list(responses or [])

    def check_model(self):
        pass

    def annotate(self, row):
        self.calls += 1
        if self.responses:
            item = self.responses.pop(0)
            if isinstance(item, BaseException):
                raise item
            return item
        return response(annotation())


def response(a):
    return {
        "http_status": 200,
        "error": None,
        "body": json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(a)}}]}),
    }


def test_pilot_selection_deterministic_mixed_train_only(dataset):
    cfg, manifest, _ = dataset
    chosen = prepare.select_pilot(manifest, cfg["seed"])
    shuffled = {**manifest, "records": list(reversed(manifest["records"]))}
    assert chosen == prepare.select_pilot(shuffled, cfg["seed"])
    assert len(chosen) == 20 and set(chosen) <= set(manifest["splits"]["train"])
    assert len({prepare.bucket(r["instruction"]) for r in manifest["records"] if r["id"] in chosen}) == 6


@pytest.mark.parametrize("kind", ["group", "session", "pixels", "bytes", "membership"])
def test_split_leakage_prevention(dataset, kind):
    _, m, _ = dataset
    m = copy.deepcopy(m)
    if kind == "group":
        m["records"][-1]["source_group_id"] = m["records"][0]["source_group_id"]
    elif kind == "session":
        m["records"][-1]["img_id"] = m["records"][0]["img_id"]
    elif kind in ("pixels", "bytes"):
        key = "pixel_sha256" if kind == "pixels" else "sha256"
        m["records"][-1]["target_image"][key] = m["records"][0]["source_image"][key]
    else:
        m["splits"]["test"].append(m["splits"]["train"][0])
    with pytest.raises(ValueError):
        prepare.validate_splits(m)


def test_two_image_payload(dataset):
    cfg, m, _ = dataset
    body = client.payload(cfg, m["records"][0])
    images = [c for c in body["messages"][1]["content"] if c["type"] == "image_url"]
    assert len(images) == 2 and images[0] != images[1]
    assert body["response_format"]["json_schema"]["schema"] == SCHEMA
    assert body["chat_template_kwargs"]["enable_thinking"] is False


def test_cpu_end_to_end_resume_review_and_downstream_loader(dataset):
    cfg, m, out = dataset
    fake = FakeClient()
    runner.infer(cfg, client=fake)
    assert fake.calls == 20
    runner.infer(cfg, client=fake)
    assert fake.calls == 20
    path, summary = runner.compile_run(cfg)
    assert summary["number_attempted"] == summary["schema_valid_count"] == summary["transaction_valid_count"] == 20
    assert summary["classification_counts"] == {"candidate_accepted": 20}
    assert summary["human_verified_accuracy"] is None
    built = review.build(cfg, path)
    assert built["cases"] == 20
    assert "joint_local_VLM" in (Path(built["review_directory"]) / "review-data.json").read_text()
    examples, report = load_examples(out / "manifest.json", path)
    assert len(examples["train"]) == 20 and report["examples"]["val"] == 0
    # Synthetic human decisions for gate tests only, never generated by production code.
    packet = read(Path(built["review_directory"]) / "review-data.json")
    decisions = {
        r["id"]: {
            "decision": "correct",
            "reviewer": "SYNTHETIC_TEST_ONLY",
            "notes": "fixture",
            **{a: "correct" for a in AXES},
        }
        for r in packet["cases"]
    }
    csv_path = out / "synthetic-test-review.csv"
    csv_path.write_bytes(export_reviews(packet, decisions))
    assert review.audit(cfg, csv_path, path)["ready"]
    expansion = review.expand(cfg, csv_path, path)
    assert len(expansion["ids"]) == 23
    runner.infer(cfg, scope="expanded", client=fake)
    assert fake.calls == 23  # Reuses exact pilot evidence, never processes project test.
    full_path, summary = runner.compile_run(cfg, scope="expanded")
    assert summary["selected_pairs"] == 23
    examples, _ = load_examples(out / "manifest.json", full_path)
    assert len(examples["val"]) == 1
    assert review.audit(cfg, csv_path, full_path)["ready"]


def test_malformed_retry_raw_preserved_and_failure_not_silenced(dataset):
    cfg, _, out = dataset
    fake = FakeClient([{"http_status": 200, "body": "bad", "error": None}])
    runner.infer(cfg, limit=1, client=fake)
    folder = sample_dir(out, read(out / "run.json")["pilot_ids"][0])
    assert read(folder / "attempt-001.raw.json")["body"] == "bad"
    assert read(folder / "final.json")["attempts"] == 2
    assert fake.calls == 2


def test_timeout_budget_and_resume(dataset):
    cfg, _, out = dataset
    raw = {"http_status": None, "body": "", "error": "ReadTimeout"}
    fake = FakeClient([raw, raw])
    runner.infer(cfg, limit=1, client=fake)
    runner.infer(cfg, limit=1, client=fake)
    assert fake.calls == 2
    _, summary = runner.compile_run(cfg)
    assert summary["number_attempted"] == 1 and summary["successful_inference_count"] == 0
    assert summary["classification_counts"]["rejected"] == 1
    assert any("ReadTimeout" in k for k in summary["failure_review_reasons"])


def test_interrupted_inference_consumes_bounded_attempt_and_resumes(dataset):
    cfg, _, out = dataset
    fake = FakeClient([KeyboardInterrupt()])
    with pytest.raises(KeyboardInterrupt):
        runner.infer(cfg, limit=1, client=fake)
    runner.infer(cfg, limit=1, client=fake)
    folder = sample_dir(out, read(out / "run.json")["pilot_ids"][0])
    assert read(folder / "attempt-001.raw.json")["error"].startswith("interrupted")
    assert fake.calls == 2 and read(folder / "final.json")["status"] == "completed"


def test_crash_after_response_does_not_reinfer(dataset, monkeypatch):
    cfg, _, out = dataset
    original = runner.write_once

    def crash(path, data):
        if Path(path).name == "annotation.json":
            raise KeyboardInterrupt()
        return original(path, data)

    fake = FakeClient()
    monkeypatch.setattr(runner, "write_once", crash)
    with pytest.raises(KeyboardInterrupt):
        runner.infer(cfg, limit=1, client=fake)
    monkeypatch.setattr(runner, "write_once", original)
    runner.infer(cfg, limit=1, client=fake)
    assert fake.calls == 1


def test_missing_image_and_changed_provenance(dataset):
    cfg, m, out = dataset
    first = read(out / "run.json")["pilot_ids"][0]
    row = next(r for r in m["records"] if r["id"] == first)
    (prepare.ROOT / row["source_image"]["path"]).unlink()
    fake = FakeClient()
    runner.infer(cfg, limit=1, client=fake)
    assert fake.calls == 0
    _, summary = runner.compile_run(cfg)
    assert summary["classification_counts"]["rejected"] == 1
    assert summary["number_attempted"] == 0
    changed = copy.deepcopy(cfg)
    changed["generation"]["temperature"] = 0.3
    with pytest.raises(ValueError, match="provenance_changed"):
        runner.infer(changed, client=fake)


def test_raw_tampering_detected(dataset):
    cfg, _, out = dataset
    runner.infer(cfg, limit=1, client=FakeClient())
    folder = sample_dir(out, read(out / "run.json")["pilot_ids"][0])
    (folder / "attempt-001.raw.json").write_text("{}")
    with pytest.raises(ValueError, match="raw_response_evidence_changed"):
        runner.compile_run(cfg)


def test_expand_without_reviews_is_blocked(dataset):
    cfg, _, _ = dataset
    runner.infer(cfg, client=FakeClient())
    path, _ = runner.compile_run(cfg)
    built = review.build(cfg, path)
    packet = read(Path(built["review_directory"]) / "review-data.json")
    csv_path = prepare.ROOT / "unreviewed.csv"
    csv_path.write_bytes(export_reviews(packet, {}))
    assert not review.audit(cfg, csv_path, path)["ready"]
    with pytest.raises(ValueError, match="human_quality_gate_not_met"):
        review.expand(cfg, csv_path, path)


def test_incomplete_finish_and_empty_content():
    raw = response(annotation())
    body = json.loads(raw["body"])
    body["choices"][0]["finish_reason"] = "length"
    raw["body"] = json.dumps(body)
    result, a = runner.attempt_result(raw)
    assert a is None and result["error"] == "incomplete_or_multiple_completion"
    body["choices"][0].update(finish_reason="stop", message={"content": None})
    raw["body"] = json.dumps(body)
    assert runner.attempt_result(raw)[1] is None


@pytest.mark.parametrize(
    "body",
    [[], None, {"choices": "bad"}, {"choices": [None]}, {"choices": [{"message": None, "finish_reason": "stop"}]}],
)
def test_malformed_http_envelope_is_inspectable_failure(body):
    result, a = runner.attempt_result({"http_status": 200, "error": None, "body": json.dumps(body)})
    assert a is None and not result["schema_valid"] and result["error"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda c: c["server"].update(base_url="https://api.openai.com/v1"),
        lambda c: c["model"].update(revision="main"),
        lambda c: c["model"].update(trust_remote_code=True),
        lambda c: c["http"].update(max_attempts=999),
        lambda c: c.update(output="data/processed/level2/magicbrush-v1"),
    ],
)
def test_config_disallows_cloud_unpinned_model_and_old_outputs(dataset, mutation):
    cfg, _, _ = dataset
    mutation(cfg)
    path = prepare.ROOT / "invalid-config.json"
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):
        prepare.config(path)


def test_local_http_redirect_and_size_errors_preserve_body(dataset):
    cfg, m, _ = dataset
    cfg["http"]["max_response_bytes"] = 5

    class Response:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def iter_content(self, size):
            yield b"too large body"

    class Session:
        def post(self, url, **kwargs):
            assert url.startswith("http://127.0.0.1:")
            assert kwargs["allow_redirects"] is False
            assert kwargs["stream"] is True
            return Response()

    local = client.LocalClient(cfg)
    assert not local.session.trust_env
    local.session = Session()
    raw = local.annotate(m["records"][0])
    assert raw["error"] == "response_size_limit" and "too large" in raw["body"]


def test_runtime_checker_and_server_plan_do_not_require_gpu(dataset, monkeypatch, tmp_path):
    from types import SimpleNamespace

    from tools.magicbrush_joint_v1 import server

    cfg, _, _ = dataset
    args = server.server_args(cfg)
    assert "--trust-remote-code" not in args and args[args.index("--max-num-seqs") + 1] == "1"
    assert args[args.index("--limit-mm-per-prompt") + 1] == '{"image":2,"video":0}'
    monkeypatch.setattr(server.importlib.metadata, "version", lambda p: "wrong")
    with pytest.raises(ValueError, match="vllm_version_mismatch"):
        server.runtime(cfg)
    monkeypatch.setattr(server.importlib.metadata, "version", lambda p: cfg["server"]["vllm_version"])
    monkeypatch.setattr(
        server.importlib.util, "find_spec", lambda p: SimpleNamespace(origin=str(tmp_path / "__init__.py"))
    )
    files = [
        "model_executor/models/registry.py",
        "model_executor/models/qwen3_5.py",
        "model_executor/layers/quantization/compressed_tensors/compressed_tensors.py",
        "model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a4_nvfp4.py",
    ]
    for name in files:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(cfg["model"]["architecture"])
    assert len(server.runtime(cfg)["runtime_source_sha256"]) == 4
