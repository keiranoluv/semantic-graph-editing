import functools
import http.server
import json
import threading
from pathlib import Path

import pytest
from PIL import Image

from tools.level3_qwen_unified.core import atomic

playwright = pytest.importorskip("playwright.sync_api")


def test_recheck_keeps_passed_rows_and_rejects_foreign_import(tmp_path):
    Image.new("RGB", (60, 40), "green").save(tmp_path / "image.png")
    rows = [
        {
            "task_id": f"t{i}",
            "binding": f"b{i}",
            "status": "reviewed",
            "edit_fulfillment": i not in (1, 3),
            "preservation": i not in (2, 3),
        }
        for i in range(200)
    ]
    cases = [
        {
            "task_id": f"t{i}",
            "binding": f"b{i}",
            "index": i + 1,
            "image_id": f"image{i}",
            "prompt": "Synthetic edit",
            "source": {"native": "image.png", "preview": "image.png"},
            "target": {"native": "image.png", "preview": "image.png"},
        }
        for i in (1, 2, 3)
    ]
    normalized = {
        "schema": "qwen-unified-review-v1",
        "cohort_sha256": "cohort",
        "reviews": rows,
        "normalization": {"raw_sha256": "raw", "defaulted_fields": {"t199": ["preservation"]}},
    }
    atomic(tmp_path / "review-data.json", {"raw_sha256": "raw", "cases": cases, "normalized": normalized})
    source = Path(__file__).resolve().parents[2] / "tools/level3_label_recheck/viewer.html"
    (tmp_path / "comparison.html").write_text(source.read_text())
    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_path))
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with playwright.sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--disable-gpu"])
            page = browser.new_page(accept_downloads=True)
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(f"http://127.0.0.1:{server.server_port}/comparison.html")
            page.wait_for_function("document.querySelectorAll('#pick option').length===3")
            assert page.locator("[data-field]").count() == 2 and page.locator("textarea").count() == 0
            assert not any(s in page.locator("body").inner_text() for s in ("CLIPScore", "LPIPS", "Pix2Pix"))
            page.locator("[data-field=edit_fulfillment]").select_option("true")
            assert page.locator("#pick option").count() == 3  # corrected rows stay visible
            page.reload()
            page.wait_for_function("document.querySelector('[data-field=edit_fulfillment]')?.value==='true'")
            with page.expect_download() as download:
                page.locator("#export").click()
            pack = json.loads(Path(download.value.path()).read_text())
            assert len(pack["reviews"]) == 200 and pack["reviews"][1]["edit_fulfillment"] is True
            assert pack["reviews"][199] == rows[199]
            assert pack["recheck"]["changes"][0]["task_id"] == "t1"
            assert pack["normalization"]["defaulted_fields"] == normalized["normalization"]["defaulted_fields"]
            # New browser storage can import corrections; reject stale bindings atomically.
            page.evaluate("localStorage.clear()")
            page.reload()
            page.wait_for_function("document.querySelector('[data-field=edit_fulfillment]')?.value==='false'")
            page.locator("#import").set_input_files(
                {"name": "ok.json", "mimeType": "application/json", "buffer": json.dumps(pack).encode()}
            )
            page.wait_for_function("document.querySelector('[data-field=edit_fulfillment]')?.value==='true'")
            pack["reviews"][3]["binding"] = "other-image"
            page.locator("#import").set_input_files(
                {"name": "bad.json", "mimeType": "application/json", "buffer": json.dumps(pack).encode()}
            )
            page.wait_for_function("document.querySelector('#notice').textContent.includes('không khớp')")
            assert page.locator("[data-field=edit_fulfillment]").input_value() == "true"
            assert not errors
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
