"""Paired-image + external JSON workflow. All test labels are synthetic."""

import copy
import json
import threading
import zipfile
from http.server import ThreadingHTTPServer

import pytest

from tools.edit_model_v1.data import load_examples
from tools.emnlp2020_graph_modification_v1.dataset import sha256
from tools.magicbrush_manual_v1.store import Conflict
from tools.magicbrush_manual_v2.proposal import parse, prompt
from tools.magicbrush_manual_v2.server import handler
from tools.magicbrush_manual_v2.store import Store


@pytest.fixture
def legacy_store(tmp_path):
    from test_magicbrush_manual_v1 import store as fixture

    return fixture.__wrapped__(tmp_path)


@pytest.fixture
def store(legacy_store):
    return Store(
        legacy_store.manifest_path,
        legacy_store.root / "data/manual-v2",
        legacy_output=legacy_store.output,
        root=legacy_store.root,
    )


def proposal(identifier):
    return {
        "sample_id": identifier,
        "source_graph": {"nodes": [{"id": "n001", "label": "cup", "attributes": ["red"]}], "edges": []},
        "target_graph": {"nodes": [{"id": "n001", "label": "cup", "attributes": ["blue"]}], "edges": []},
        "notes": "Synthetic zero-shot response fixture only.",
    }


def test_both_images_instruction_immediately_available(store):
    case = store.case(store.review_ids[0])
    assert case["instruction"] and case["target_image"] and case["source_image"]
    assert store.image(case["id"], "target")[0]
    assert case["state"]["revision"] == 0 and not case["state"]["source_locked"]
    with pytest.raises(ValueError, match="test chưa được mở"):
        store.case("magicbrush:21:1")


def test_snapshot_preserves_every_v1_event_and_marker(legacy_store):
    identifier = legacy_store.review_ids[0]
    old = legacy_store.save(
        identifier,
        "lock_source",
        {
            "base_revision": 0,
            "source_graph": proposal(identifier)["source_graph"],
            "source_points": {"n001": {"x": 0.3, "y": 0.4}},
            "annotator": "UNIT_TEST_ONLY",
        },
    )["state"]
    original = {str(p): sha256(p) for p in legacy_store.output.rglob("*.json")}
    current = Store(
        legacy_store.manifest_path,
        legacy_store.root / "data/manual-v2",
        legacy_output=legacy_store.output,
        root=legacy_store.root,
    )
    migrated = current.latest(identifier)
    assert migrated["source_graph"] == old["source_graph"] and migrated["target_graph"] == old["target_graph"]
    assert migrated["source_points"] == old["source_points"]
    assert migrated["legacy_event_sha256"] == old["event_sha256"]
    assert {str(p): sha256(p) for p in legacy_store.output.rglob("*.json")} == original
    current.save(identifier, "draft", {"base_revision": 0, "notes": "v2 edit"})
    assert {str(p): sha256(p) for p in legacy_store.output.rglob("*.json")} == original
    fresh = Store(legacy_store.manifest_path, current.output, legacy_output=legacy_store.output, root=legacy_store.root)
    assert fresh.latest(identifier)["notes"] == "v2 edit"


def test_zero_shot_prompt_has_no_existing_annotations(store):
    identifier = store.review_ids[0]
    store.save(
        identifier,
        "draft",
        {
            "base_revision": 0,
            "source_graph": {"nodes": [{"id": "n001", "label": "PRIOR_SECRET_LABEL", "attributes": []}], "edges": []},
        },
    )
    text = prompt(store.records[identifier])
    assert identifier in text and "Image 1 = SOURCE" in text and "Image 2 = TARGET" in text
    assert "PRIOR_SECRET_LABEL" not in text and '"nodes": []' in text


def test_preview_is_readonly_and_import_is_pending(store):
    identifier = store.review_ids[0]
    text = json.dumps(proposal(identifier))
    preview = parse(text, identifier)
    assert preview["validation"]["valid"] and preview["operation_count"] == 1
    assert store.latest(identifier)["revision"] == 0
    result = store.import_json(identifier, "```json\n" + text + "\n```", 0, "UNIT_TEST_ONLY")
    assert result["state"]["status"] == "draft"
    assert result["state"]["annotation_origin"] == "external_json_proposal"
    assert not result["state"]["instruction_matches"] and not result["state"]["unchanged_checked"]
    assert store.compile()["report"]["completed"] == 0
    with pytest.raises(ValueError, match="đối chiếu"):
        store.save(identifier, "complete", {"base_revision": 1})


def test_checked_proposal_exports_with_origin_and_source_only_input(store):
    identifier = store.review_ids[0]
    store.import_json(identifier, json.dumps(proposal(identifier)), 0, "UNIT_TEST_ONLY")
    store.save(identifier, "complete", {"base_revision": 1, "instruction_matches": True, "unchanged_checked": True})
    bundle = store.compile()
    assert bundle["report"]["completed"] == 1 and not bundle["report"]["ready_for_training"]
    assert bundle["edits"]["records"][0]["label_origin"] == "external_json_proposal"
    assert bundle["edits"]["records"][0]["human_review_status"] == "pending"
    directory = store.output / "exports" / bundle["report"]["snapshot"]
    examples, _ = load_examples(directory / "manifest.json", directory / "edits.json")
    assert set(examples["train"][0].model_input()) == {"source", "instruction"}
    assert bundle == store.compile()


@pytest.mark.parametrize("fault", ["syntax", "sample_id", "duplicate_keys", "dangling", "unknown", "nan"])
def test_bad_import_never_overwrites(store, fault):
    identifier = store.review_ids[0]
    text = json.dumps(proposal(identifier))
    if fault == "syntax":
        text = "{"
    elif fault == "sample_id":
        text = text.replace(identifier, "wrong-id")
    elif fault == "duplicate_keys":
        text = '{"sample_id":"a","sample_id":"b"}'
    elif fault == "nan":
        text = '{"sample_id":NaN}'
    else:
        data = proposal(identifier)
        if fault == "unknown":
            data["operations"] = []
        else:
            data["target_graph"]["edges"] = [
                {"id": "newe001", "subject": "n001", "predicate": "on", "object": "missing"}
            ]
        text = json.dumps(data)
    with pytest.raises(ValueError):
        store.import_json(identifier, text, 0)
    assert store.latest(identifier)["revision"] == 0


def test_import_conflict_keeps_original_and_source_can_be_edited(store):
    identifier = store.review_ids[0]
    store.save(identifier, "draft", {"base_revision": 0, "notes": "other tab"})
    with pytest.raises(Conflict):
        store.import_json(identifier, json.dumps(proposal(identifier)), 0)
    assert store.latest(identifier)["notes"] == "other tab"
    source = proposal(identifier)["source_graph"]
    store.save(identifier, "draft", {"base_revision": 1, "source_graph": source})
    source = copy.deepcopy(source)
    source["nodes"][0]["label"] = "bowl"
    store.save(identifier, "draft", {"base_revision": 2, "source_graph": source})
    assert store.latest(identifier)["source_graph"]["nodes"][0]["label"] == "bowl"


def test_import_long_combined_notes_keeps_current_revision(store):
    identifier = store.review_ids[0]
    store.save(identifier, "draft", {"base_revision": 0, "notes": "a" * 4000})
    with pytest.raises(ValueError, match="4000"):
        store.import_json(identifier, json.dumps(proposal(identifier)), 1)
    assert store.latest(identifier)["revision"] == 1
    assert store.latest(identifier)["notes"] == "a" * 4000


def test_arbitrary_ids_normalize_without_target_derived_source_identity(store):
    identifier = store.review_ids[0]
    data = proposal(identifier)
    data["source_graph"]["nodes"][0]["id"] = "original-cup"
    data["target_graph"]["nodes"][0]["id"] = "original-cup"
    first = parse(json.dumps(data), identifier)
    data["target_graph"]["nodes"][0]["attributes"] = ["green"]
    second = parse(json.dumps(data), identifier)
    assert first["source_graph"] == second["source_graph"]
    assert first["source_graph"]["nodes"][0]["id"] == "n001"


@pytest.fixture
def server(store):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler(store))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}", store
    httpd.shutdown()
    httpd.server_close()
    thread.join()


@pytest.fixture
def browser():
    playwright = pytest.importorskip("playwright.sync_api")
    with playwright.sync_playwright() as runtime:
        instance = runtime.chromium.launch(args=["--disable-gpu"])
        yield instance
        instance.close()


def test_browser_pair_visible_zip_prompt_paste_and_complete(browser, server, tmp_path):
    url, store = server
    page = browser.new_page(viewport={"width": 1512, "height": 1080}, accept_downloads=True)
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(url)
    page.wait_for_function(
        "() => document.getElementById('source-image').naturalWidth>0 && document.getElementById('target-image').naturalWidth>0"
    )
    identifier = page.locator(".queue-item.active").get_attribute("data-id")
    assert page.locator("#instruction").inner_text() == store.records[identifier]["instruction"]
    assert page.locator("#source-card").is_visible() and page.locator("#target-card").is_visible()
    assert "Xác nhận nguồn" not in page.locator("#primary").inner_text()
    with page.expect_download() as download:
        page.locator("#download-pair").click()
    path = tmp_path / "pair.zip"
    download.value.save_as(path)
    with zipfile.ZipFile(path) as archive:
        assert archive.read("1-source.png") == store.image(identifier, "source")[0]
        assert archive.read("2-target.png") == store.image(identifier, "target")[0]
        assert identifier in archive.read("prompt.txt").decode()
    page.locator("#copy-prompt").click()
    page.wait_for_function(
        "() => document.getElementById('notice').textContent.includes('Đã sao chép prompt') || document.getElementById('prompt-dialog').open"
    )
    if page.locator("#prompt-dialog").is_visible():
        assert identifier in page.locator("#prompt-text").input_value()
        page.locator("#prompt-close").click()
    page.locator("#annotator").fill("UNIT_TEST_ONLY")
    page.locator("#json-open").click()
    page.locator("#proposal-text").fill(json.dumps(proposal(identifier)))
    assert page.locator("#json-apply").is_disabled()
    page.locator("#json-preview").click()
    page.wait_for_function("() => !document.getElementById('json-apply').disabled")
    assert "1 thao tác" in page.locator("#json-result").inner_text()
    page.locator("#json-apply").click()
    page.wait_for_selector("#json-dialog", state="hidden")
    assert page.get_by_label("Thuộc tính nguồn 1", exact=True).input_value() == "red"
    assert page.get_by_label("Thuộc tính đích 1", exact=True).input_value() == "blue"
    assert not page.get_by_label("Tên đối tượng nguồn 1", exact=True).is_disabled()
    assert store.latest(identifier)["status"] == "draft"
    page.locator("#primary").click()
    page.locator("#instruction-matches").check()
    page.locator("#unchanged-checked").check()
    page.locator("#primary").click()
    page.wait_for_function("() => document.getElementById('total-progress').textContent.startsWith('1 /')")
    assert store.latest(identifier)["status"] == "completed" and not errors
    page.close()


def test_browser_json_file_error_and_mobile(browser, server, tmp_path):
    url, store = server
    page = browser.new_page(viewport={"width": 390, "height": 844})
    page.goto(url)
    page.wait_for_selector(".queue-item.active")
    identifier = page.locator(".queue-item.active").get_attribute("data-id")
    assert page.locator("#target-card").is_visible()
    assert page.evaluate("() => document.documentElement.scrollWidth<=innerWidth")
    page.locator("#json-open").click()
    page.locator("#proposal-text").fill("{")
    page.locator("#json-preview").click()
    page.wait_for_function("() => document.getElementById('json-result').classList.contains('error')")
    assert page.locator("#json-apply").is_disabled() and store.latest(identifier)["revision"] == 0
    path = tmp_path / "proposal.json"
    path.write_text(json.dumps(proposal(identifier)))
    page.locator("#proposal-file").set_input_files(path)
    page.locator("#json-preview").click()
    page.wait_for_function("() => !document.getElementById('json-apply').disabled")
    page.locator("#json-apply").click()
    page.wait_for_selector("#json-dialog", state="hidden")
    assert store.latest(identifier)["annotation_origin"] == "external_json_proposal"
    page.close()
