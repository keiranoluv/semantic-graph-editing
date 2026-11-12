"""Browser review QA with clearly synthetic temporary fixtures, no GPU computation."""

import csv
import io
import json
import threading
from http.server import ThreadingHTTPServer

import pytest
from PIL import Image

from tools import review_magicbrush as review

playwright = pytest.importorskip("playwright.sync_api")


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setattr(review, "ROOT", tmp_path)
    output = tmp_path / "reviews/fixture"
    output.mkdir(parents=True)
    images = tmp_path / "data/images"
    images.mkdir(parents=True)
    Image.new("RGB", (80, 60), "red").save(images / "source.png")
    Image.new("RGB", (80, 60), "blue").save(images / "target.png")
    src = {"nodes": [{"id": "n001", "label": "cup", "attributes": ["red"]}], "edges": []}
    dst = {"nodes": [{"id": "n001", "label": "cup", "attributes": ["blue"]}], "edges": []}
    label = {
        "status": "candidate",
        "source_graph": src,
        "target_graph": dst,
        "problems": [],
        "source_boxes": {"n001": [100, 100, 900, 900]},
        "target_boxes": {"n001": [100, 100, 900, 900]},
        "operations": [{"kind": "replace_node", "before": src["nodes"][0], "after": dst["nodes"][0]}],
    }
    cases = []
    for index in (0, 1):
        cases.append(
            {
                "id": f"synthetic:{index}",
                "img_id": str(index),
                "turn_index": 1,
                "split": "train",
                "source_group_id": "fixture",
                "instruction": "Synthetic fixture: change the cup from red to blue.",
                "source_image": {"path": "data/images/source.png", "width": 80, "height": 60, "sha256": "s"},
                "target_image": {"path": "data/images/target.png", "width": 80, "height": 60, "sha256": "t"},
                "evidence_sha256": f"fixture-{index}",
                "label": label if index == 0 else {"status": "pending_graphs"},
                "graphs_ready": index == 0,
            }
        )
    packet = {
        "cases": cases,
        "axes": review.AXES,
        "export_fields": review.FIELDS,
        "packet_sha256": "synthetic-fixture-only",
    }
    data = json.dumps(packet).replace("<", "\\u003c")
    (output / "index.html").write_text((review.TEMPLATES / "index.html").read_text().replace("__REVIEW_DATA__", data))
    for name in ("viewer.js", "viewer.css"):
        (output / name).write_bytes((review.TEMPLATES / name).read_bytes())
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), review.handler(output, packet))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}", output
    httpd.shutdown()
    httpd.server_close()
    thread.join()


@pytest.fixture
def browser():
    with playwright.sync_playwright() as runtime:
        try:
            instance = runtime.chromium.launch(args=["--disable-gpu"])
        except playwright.Error as error:
            pytest.skip(str(error))
        yield instance
        instance.close()


def test_actual_review_interactions_roundtrip_and_pending_graphs_cannot_be_approved(browser, server, tmp_path):
    url, output = server
    context = browser.new_context(accept_downloads=True)
    page = context.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(url)
    assert page.locator("#progress").inner_text().startswith("0/2")
    assert page.locator(".graph-svg").count() == 2 and page.locator("#decision").input_value() == ""
    page.locator("#decision").select_option("correct")
    page.locator("#export").click()
    assert "cần tên" in page.locator("#message").inner_text()
    for axis in review.AXES:
        page.locator("#axis-" + axis).select_option("correct")
    page.locator("#reviewer").fill("Fixture QA")
    notes = 'Đúng, "mẫu"\ndòng 2'
    page.locator("#notes").fill(notes)
    with page.expect_download() as downloaded:
        page.locator("#export").click()
    target = tmp_path / "review.csv"
    downloaded.value.save_as(target)
    rows = list(csv.DictReader(io.StringIO(target.read_text(encoding="utf-8-sig"))))
    assert rows[0]["notes"] == notes and rows[0]["decision"] == "correct" and rows[1]["decision"] == ""
    page.locator("#save").click()
    page.wait_for_function("document.getElementById('message').textContent.startsWith('Đã lưu')")
    assert len(list((output / "submissions").glob("*.csv"))) == 1
    report = json.loads(next((output / "submissions").glob("*.report.json")).read_text())
    assert report["reviewed_pairs"] == 1 and report["automatic_approval"] is False
    page.locator("#next").click()
    assert page.locator("#decision").is_disabled()
    assert "CHỜ SINH ĐỒ THỊ" in page.locator("#status").inner_text()
    assert page.locator(".graph-svg").count() == 0
    page.evaluate("localStorage.clear()")
    page.reload()
    page.locator("#import").set_input_files(target)
    assert "Đã nhập CSV" in page.locator("#message").inner_text()
    assert page.locator("#progress").inner_text().startswith("1/2")
    assert not errors
    context.close()


def test_mobile_file_viewer_loads_images_without_server(browser, server):
    _, output = server
    context = browser.new_context(viewport={"width": 390, "height": 844})
    page = context.new_page()
    page.goto((output / "index.html").as_uri())
    page.wait_for_function("[...document.querySelectorAll('.photo img')].every(i=>i.complete&&i.naturalWidth>0)")
    assert page.locator("#save").is_disabled()
    assert page.evaluate("document.documentElement.scrollWidth<=innerWidth")
    context.close()
