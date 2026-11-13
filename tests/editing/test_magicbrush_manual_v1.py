"""CPU-only manual annotation regressions; all written labels are synthetic."""

import copy
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest
from PIL import Image

from src.editing.operations import EditGraph, Operation, apply_operations
from tools.edit_model_v1.data import load_examples
from tools.emnlp2020_graph_modification_v1.dataset import read, sha256, write
from tools.magicbrush_manual_v1.server import handler
from tools.magicbrush_manual_v1.store import Conflict, Store, validate_pair


def g(label="cup", color="red"):
    return {"nodes": [{"id": "n001", "label": label, "attributes": [color]}], "edges": []}


@pytest.fixture
def store(tmp_path):
    root = tmp_path / "project"
    records = []
    for i in range(22):
        split = "train" if i < 20 else "val" if i == 20 else "test"
        row = {
            "id": f"magicbrush:{i}:1",
            "img_id": str(i),
            "turn_index": 1,
            "split": split,
            "official_split": "train",
            "source_group_id": f"group:{i}",
            "instruction": "Change the cup from red to blue. Synthetic fixture only.",
        }
        for role in ("source", "target"):
            path = root / "data" / f"{i}-{role}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            # Different files/pixels across splits: no artificial leakage bypass.
            Image.new("RGB", (160, 120), (i * 10, 50, 40 if role == "source" else 200)).save(path)
            row[role + "_image"] = {
                "path": str(path.relative_to(root)),
                "sha256": sha256(path),
                "pixel_sha256": f"fixture-pixels-{i}-{role}",
                "width": 160,
                "height": 120,
                "bytes": path.stat().st_size,
            }
        records.append(row)
    manifest = {
        "dataset": "osunlp/MagicBrush",
        "revision": "synthetic-unit-fixture-only",
        "records": records,
        "review_ids": [r["id"] for r in records[:20]],
        "splits": {s: [r["id"] for r in records if r["split"] == s] for s in ("train", "val", "test")},
    }
    path = root / "data" / "manifest.json"
    write(path, manifest)
    return Store(path, root / "data" / "manual", root=root)


def save(store, identifier, action="draft", **changes):
    return store.save(identifier, action, {"base_revision": store.latest(identifier)["revision"], **changes})


def locked(store, identifier=None):
    identifier = identifier or store.review_ids[0]
    return save(
        store,
        identifier,
        "lock_source",
        source_graph=g(),
        source_points={"n001": {"x": 0.5, "y": 0.5}},
        annotator="UNIT_TEST_ONLY",
    )


def complete(store, identifier=None):
    identifier = identifier or store.review_ids[0]
    locked(store, identifier)
    return save(
        store, identifier, "complete", target_graph=g(color="blue"), instruction_matches=True, unchanged_checked=True
    )


def test_source_only_protocol_and_test_hidden(store):
    bootstrap = store.bootstrap()
    assert len(bootstrap["cases"]) == 21 and len(bootstrap["review_ids"]) == 20
    case = store.case(store.review_ids[0])
    assert case["instruction"] is case["target_image"] is None
    with pytest.raises(ValueError, match="xác nhận graph nguồn"):
        store.image(case["id"], "target")
    with pytest.raises(ValueError, match="test chưa được mở"):
        store.case("magicbrush:21:1")
    assert store.bootstrap()["counts"] == {"draft": 21}


def test_incomplete_draft_kept_and_missing_name_blocks_lock(store):
    identifier = store.review_ids[0]
    result = save(store, identifier, source_graph=g(label=""))
    assert result["state"]["source_graph"]["nodes"][0]["label"] == ""
    with pytest.raises(ValueError):
        save(store, identifier, "lock_source")
    assert store.latest(identifier)["revision"] == 1
    with pytest.raises(ValueError):
        save(store, identifier, "lock_source", annotator="UNIT_TEST_ONLY")
    assert store.latest(identifier)["revision"] == 1


def test_source_freeze_reveal_and_immutability(store):
    identifier = store.review_ids[0]
    result = locked(store)
    assert result["instruction"] and result["target_image"]
    assert result["state"]["target_graph"] == result["state"]["source_graph"]
    assert result["state"]["target_points"] == {}
    with pytest.raises(ValueError, match="đã khóa"):
        save(store, identifier, source_graph=g(color="green"))
    with pytest.raises(ValueError, match="đã khóa"):
        save(store, identifier, source_points={})
    assert store.latest(identifier)["revision"] == 1


def test_canonical_source_uses_no_target_and_remaps_points(store):
    source = g()
    source["nodes"][0]["id"] = "n009"
    result = save(
        store,
        store.review_ids[0],
        "lock_source",
        source_graph=source,
        source_points={"n009": {"x": 0.25, "y": 0.75}},
        annotator="UNIT_TEST_ONLY",
    )
    assert result["state"]["source_graph"] == g()
    assert result["state"]["source_points"] == {"n001": {"x": 0.25, "y": 0.75}}


def test_complete_transaction_and_resume_without_duplicate_events(store):
    result = complete(store)
    assert result["state"]["status"] == "completed"
    validation = result["validation"]
    assert validation["valid"] and validation["operation_counts"]["replace_node"] == 1
    ops = [Operation.from_dict(o) for o in validation["operations"]]
    assert apply_operations(EditGraph.from_dict(validation["source_graph"]), ops) == EditGraph.from_dict(
        validation["target_graph"]
    )
    fresh = Store(store.manifest_path, store.output, root=store.root)
    assert fresh.latest(result["id"]) == result["state"]
    again = save(fresh, result["id"])
    assert again["state"]["status"] == "completed" and again["state"]["revision"] == 2
    save(fresh, result["id"], target_graph=g(color="green"))
    assert fresh.latest(result["id"])["status"] == "source_locked"
    assert len(fresh.history(result["id"])) == 3


def test_finalization_requires_image_checks_and_real_change(store):
    locked(store)
    identifier = store.review_ids[0]
    with pytest.raises(ValueError, match="hai mục"):
        save(store, identifier, "complete", target_graph=g(color="blue"))
    with pytest.raises(ValueError, match="chưa có thay đổi"):
        save(store, identifier, "complete", instruction_matches=True, unchanged_checked=True)
    assert store.latest(identifier)["status"] == "source_locked"


def test_defer_and_reopen_preserve_history(store):
    result = complete(store)
    identifier = result["id"]
    result = save(store, identifier, "reopen_source")
    assert result["state"]["target_graph"] is None and not result["state"]["source_locked"]
    assert result["instruction"] is None
    assert store.history(identifier)[1]["status"] == "completed"
    with pytest.raises(ValueError, match="lý do"):
        save(store, identifier, "defer")
    result = save(store, identifier, "defer", reason="Unclear image, synthetic test.")
    assert result["state"]["status"] == "deferred" and store.compile()["report"]["completed"] == 0


def test_optimistic_conflict_and_event_tamper(store):
    identifier = store.review_ids[0]
    save(store, identifier, notes="first")
    with pytest.raises(Conflict):
        store.save(identifier, "draft", {"base_revision": 0, "notes": "stale"})
    assert store.latest(identifier)["notes"] == "first"
    path = next(store.directory(identifier).glob("*.json"))
    event = read(path)
    event["notes"] = "tampered"
    path.write_text(json.dumps(event))
    with pytest.raises(ValueError, match="Lịch sử"):
        store.latest(identifier)


def test_image_hash_and_manifest_changes_fail_closed(store):
    identifier = store.review_ids[0]
    path = store.root / store.records[identifier]["source_image"]["path"]
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        store.image(identifier, "source")
    manifest = read(store.manifest_path)
    manifest["revision"] = "changed"
    store.manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Manifest gốc"):
        store.bootstrap()


@pytest.mark.parametrize("change", ["dangling", "duplicate", "new_identity", "noop"])
def test_invalid_labels_rejected_without_changing_semantics(change):
    source, target = g(), g(color="blue")
    if change == "dangling":
        target["edges"] = [{"id": "newe001", "subject": "n001", "predicate": "on", "object": "missing"}]
    elif change == "duplicate":
        target["edges"] = [
            {"id": f"newe{i:03d}", "subject": "n001", "predicate": "near", "object": "n001"} for i in (1, 2)
        ]
    elif change == "new_identity":
        target["nodes"].append({"id": "n999", "label": "dog", "attributes": []})
    else:
        target = copy.deepcopy(source)
    assert not validate_pair(source, target)["valid"]


def test_node_deletion_requires_explicit_incident_edge_diff():
    source = {
        "nodes": [
            {"id": "n001", "label": "person", "attributes": []},
            {"id": "n002", "label": "cup", "attributes": ["red"]},
        ],
        "edges": [{"id": "e001", "subject": "n001", "predicate": "holding", "object": "n002"}],
    }
    target = {"nodes": [source["nodes"][0]], "edges": []}
    result = validate_pair(source, target)
    assert (
        result["valid"] and result["operation_counts"]["delete_node"] == result["operation_counts"]["delete_edge"] == 1
    )


def test_snapshot_existing_loader_and_review_packet(store, monkeypatch):
    from tools import review_magicbrush

    complete(store)
    bundle = store.compile()
    directory = store.output / "exports" / bundle["report"]["snapshot"]
    examples, _ = load_examples(directory / "manifest.json", directory / "edits.json")
    assert len(examples["train"]) == 1 and examples["train"][0].model_input().keys() == {"source", "instruction"}
    label = bundle["edits"]["records"][0]
    assert label["label_origin"] == "human_image_annotation" and label["human_review_status"] == "pending"
    assert not bundle["report"]["ready_for_training"]
    assert bundle == store.compile()
    monkeypatch.setattr(review_magicbrush, "ROOT", store.root)
    packet = review_magicbrush.make_packet(directory / "manifest.json", directory / "edits.json")
    assert len(packet["cases"]) == 20 and sum(r["graphs_ready"] for r in packet["cases"]) == 1
    assert packet["human_reviews_created"] == 0


@pytest.fixture
def server(store):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler(store))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}", store
    httpd.shutdown()
    httpd.server_close()
    thread.join()


def test_api_rejects_foreign_origin_and_forbidden_paths(server):
    url, store = server
    request = urllib.request.Request(
        url + "/api/export", data=b"{}", headers={"Origin": "http://evil.example", "Content-Type": "application/json"}
    )
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(request)
    assert error.value.code == 403
    with pytest.raises(urllib.error.HTTPError):
        urllib.request.urlopen(url + "/../../AGENTS.md")
    identifier = store.review_ids[0]
    with pytest.raises(urllib.error.HTTPError):
        urllib.request.urlopen(url + "/api/image?id=" + identifier + "&role=target")


@pytest.fixture
def browser():
    playwright = pytest.importorskip("playwright.sync_api")
    with playwright.sync_playwright() as runtime:
        instance = runtime.chromium.launch(args=["--disable-gpu"])
        yield instance
        instance.close()


def wait_saved(page):
    page.wait_for_function("() => document.getElementById('save-state').textContent.startsWith('Đã lưu vào máy')")


def test_browser_full_workflow_autosave_identity_and_export(browser, server, tmp_path):
    url, store = server
    page = browser.new_page(viewport={"width": 1512, "height": 1080}, accept_downloads=True)
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(url)
    page.wait_for_function("() => document.getElementById('source-image').naturalWidth > 0")
    assert page.locator(".queue-item").count() == 20
    identifier = page.locator(".queue-item.active").get_attribute("data-id")
    assert page.locator("#target-card").is_hidden()
    page.locator("#annotator").fill("UNIT_TEST_ONLY")
    page.locator("#source-image").click(position={"x": 40, "y": 40})
    page.get_by_label("Tên đối tượng nguồn 1", exact=True).fill("person")
    page.locator("#source-quick-add").click()
    page.get_by_label("Tên đối tượng nguồn 2", exact=True).fill("cup")
    page.get_by_label("Thuộc tính nguồn 2", exact=True).fill("red")
    page.locator("#source-add-edge").click()
    page.get_by_label("Quan hệ → nguồn e001", exact=True).fill("holding")
    page.locator("#save").click()
    wait_saved(page)
    # Continue typing after server state replacement: edits must not hit stale closures.
    page.get_by_label("Thuộc tính nguồn 2", exact=True).fill("red, ceramic")
    page.locator("#save").click()
    wait_saved(page)
    assert store.latest(identifier)["source_graph"]["nodes"][1]["attributes"] == ["red", "ceramic"]
    page.locator("#primary").click()
    page.wait_for_selector("#target-card", state="visible")
    assert page.get_by_label("Tên đối tượng nguồn 1", exact=True).is_disabled()
    assert page.get_by_label("Tên đối tượng đích 2", exact=True).input_value() == "cup"
    page.get_by_label("Thuộc tính đích 2", exact=True).fill("blue, ceramic")
    page.locator("#primary").click()
    page.wait_for_selector("#check-card", state="visible")
    page.wait_for_function("() => document.getElementById('validation-badge').textContent.includes('Hợp lệ')")
    assert "Thay nhãn" in page.locator("#operations").inner_text()
    page.locator("#instruction-matches").check()
    page.locator("#unchanged-checked").check()
    page.locator("#primary").click()
    page.wait_for_function("() => document.getElementById('total-progress').textContent.startsWith('1 /')")
    assert store.latest(identifier)["status"] == "completed"
    with page.expect_download() as download:
        page.locator("#export").click()
    path = tmp_path / "fixture-download.json"
    download.value.save_as(path)
    bundle = read(path)
    assert bundle["report"]["completed"] == 1 and not bundle["report"]["independent_review_completed"]
    assert not errors
    page.close()


def test_browser_undo_cascade_and_mobile_layout(browser, server):
    url, store = server
    identifier = store.bootstrap()["cases"][0]["id"]
    source = {
        "nodes": [
            {"id": "n001", "label": "person", "attributes": []},
            {"id": "n002", "label": "cup", "attributes": ["red"]},
        ],
        "edges": [{"id": "e001", "subject": "n001", "predicate": "holding", "object": "n002"}],
    }
    save(store, identifier, "lock_source", source_graph=source, annotator="UNIT_TEST_ONLY")
    page = browser.new_page(viewport={"width": 390, "height": 844})
    page.on("dialog", lambda dialog: dialog.accept())
    page.goto(url)
    page.wait_for_selector("#target-card", state="visible")
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    page.get_by_label("Xóa đối tượng đích 2", exact=True).click()
    assert page.locator("#target-nodes .node-row").count() == 1
    assert page.locator("#target-edges .edge-row").count() == 0
    page.locator("#undo").click()
    assert page.locator("#target-nodes .node-row").count() == 2
    assert page.locator("#target-edges .edge-row").count() == 1
    page.locator("#redo").click()
    page.locator("#save").click()
    wait_saved(page)
    assert store.case(identifier)["validation"]["operation_counts"]["delete_edge"] == 1
    assert store.latest(identifier)["source_graph"] == source
    assert not page.evaluate("document.documentElement.scrollWidth > innerWidth")
    page.close()


def test_browser_recover_local_draft_without_false_completion(browser, server):
    url, store = server
    page = browser.new_page(viewport={"width": 1440, "height": 1000})
    page.goto(url)
    page.wait_for_selector("#source-quick-add")
    identifier = page.locator(".queue-item.active").get_attribute("data-id")
    # Reproduce an interrupted autosave without creating annotation data on real images.
    workspace = store.bootstrap()["workspace_sha256"]
    state = store.latest(identifier)
    local = {
        "base_revision": 0,
        "source_locked": False,
        "annotator": "UNIT_TEST_ONLY",
        **{
            k: state[k]
            for k in (
                "source_graph",
                "target_graph",
                "source_points",
                "target_points",
                "notes",
                "reason",
                "instruction_matches",
                "unchanged_checked",
            )
        },
    }
    local["source_graph"] = g()
    page.evaluate(
        "([key,value])=>localStorage.setItem(key,JSON.stringify(value))",
        [f"magicbrush-manual:{workspace}:{identifier}", local],
    )
    page.reload()
    page.get_by_role("button", name="Khôi phục bản nháp", exact=True).click()
    wait_saved(page)
    assert page.get_by_label("Tên đối tượng nguồn 1", exact=True).input_value() == "cup"
    assert store.latest(identifier)["source_graph"] == g()
    assert store.latest(identifier)["status"] == "draft"
    page.close()


def test_browser_conflict_keeps_both_drafts(browser, server):
    url, store = server
    page = browser.new_page(viewport={"width": 1440, "height": 1000})
    page.goto(url)
    page.wait_for_selector(".queue-item.active")
    identifier = page.locator(".queue-item.active").get_attribute("data-id")
    save(store, identifier, notes="Other annotator saved first.")
    page.locator("#source-quick-add").click()
    page.get_by_label("Tên đối tượng nguồn 1", exact=True).fill("dog")
    page.locator("#save").click()
    page.get_by_role("button", name="Tải lại bản trên máy chủ", exact=True).wait_for()
    assert store.latest(identifier)["notes"] == "Other annotator saved first."
    assert page.get_by_label("Tên đối tượng nguồn 1", exact=True).input_value() == "dog"
    assert page.locator("#primary").is_disabled()
    assert page.evaluate(
        "() => Object.keys(localStorage).some(k=>k.startsWith('magicbrush-manual:') && localStorage.getItem(k).includes('dog'))"
    )
    page.on("dialog", lambda d: d.accept())
    page.close()


def test_browser_import_matching_backup_is_still_draft(browser, server, tmp_path):
    url, store = server
    page = browser.new_page(viewport={"width": 1440, "height": 1000})
    page.on("dialog", lambda d: d.accept())
    page.goto(url)
    page.wait_for_selector(".queue-item.active")
    identifier = page.locator(".queue-item.active").get_attribute("data-id")
    state = store.latest(identifier)
    backup = {
        **state,
        "workspace_sha256": store.bootstrap()["workspace_sha256"],
        "base_revision": 0,
        "annotator": "UNIT_TEST_ONLY",
        "source_graph": g(),
    }
    path = tmp_path / "backup.json"
    path.write_text(json.dumps(backup))
    page.locator("#restore-file").set_input_files(path)
    page.wait_for_function("() => document.getElementById('notice').textContent.includes('Đã khôi phục bản nháp')")
    assert page.get_by_label("Tên đối tượng nguồn 1", exact=True).input_value() == "cup"
    assert store.latest(identifier)["status"] == "draft"
    page.close()
