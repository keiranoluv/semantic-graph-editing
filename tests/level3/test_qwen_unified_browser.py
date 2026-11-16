"""Optional Chromium CPU checks: 200-case review and safe label migration, no model inference."""

import functools
import http.server
import json
import threading
from pathlib import Path

import pytest
from PIL import Image

from tools.level3_qwen_unified.core import atomic

playwright = pytest.importorskip("playwright.sync_api")


def test_200_case_browser_migration_update_and_export(tmp_path):
    image = tmp_path / "test.jpg"
    Image.new("RGB", (64, 48), "green").save(image)
    asset = {"preview": "test.jpg", "native": "test.jpg"}
    cases = [
        {
            "task_id": f"t{i}",
            "index": i + 1,
            "image_id": f"source-{i % 100}",
            "prompt": f"Synthetic edit {i}",
            "kind": "add",
            "status": "ok" if i < 100 else "pending",
            "source": asset,
            "target": asset if i < 100 else None,
            "binding": f"binding-{i}" if i < 100 else None,
        }
        for i in range(200)
    ]
    data = {
        "cohort_sha256": "test-only",
        "successful": 100,
        "cases": cases,
        "legacy": [{"results_sha256": "old-results", "bindings": {"t0": "binding-0"}}],
    }
    atomic(tmp_path / "review-data.json", data)
    source = Path(__file__).resolve().parents[2] / "tools/level3_qwen_unified/viewer.html"
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
            url = f"http://127.0.0.1:{server.server_port}/comparison.html"
            page.goto(url)
            page.wait_for_function("document.querySelectorAll('#pick option').length===200")
            assert page.locator("[data-field]").count() == 2
            assert page.locator("textarea").count() == 0
            assert "Pix2Pix" not in page.locator("body").inner_text()
            assert page.locator("#count").inner_text() == "Đã có ảnh 100/200 · Đã chấm 0/200"
            # Simulate old review on the same browser origin (different ports need JSON import).
            page.evaluate(
                "localStorage.setItem('qwen-only-review-v2:old-results', JSON.stringify([{task_id:'t0',model:'qwen21',edit_fulfillment:true,preservation:false}]))"
            )
            page.reload()
            page.wait_for_function("document.querySelector('[data-field=preservation]')?.value==='false'")
            page.locator("#pick").select_option("100")
            assert page.locator("[data-field]:disabled").count() == 2
            # Add completed evidence; existing labels stay attached to original image hashes.
            data["cases"][100].update(status="ok", target=asset, binding="binding-100")
            data["successful"] = 101
            atomic(tmp_path / "review-data.json", data)
            page.locator("#refresh").click()
            page.wait_for_function("document.querySelector('#count').textContent.includes('101/200')")
            assert page.locator("[data-field=preservation]").input_value() == "false"
            page.locator("#pick").select_option("100")
            page.locator("[data-field=edit_fulfillment]").select_option("false")
            page.locator("[data-field=preservation]").select_option("true")
            with page.expect_download() as download:
                page.locator("#export").click()
            pack = json.loads(Path(download.value.path()).read_text())
            assert len(pack["reviews"]) == 200
            assert pack["reviews"][0]["preservation"] is False
            assert pack["reviews"][100]["edit_fulfillment"] is False
            assert pack["reviews"][101]["preservation"] is None
            # Wrong image binding must reject the entire import without clearing valid labels.
            pack["reviews"][100]["binding"] = "a-different-target"
            bad = tmp_path / "bad-labels.json"
            bad.write_text(json.dumps(pack))
            page.locator("#import").set_input_files(str(bad))
            page.wait_for_function("document.querySelector('#notice').textContent.includes('không khớp')")
            assert page.locator("[data-field=edit_fulfillment]").input_value() == "false"
            # A new target is unreviewed, while the old label version remains stored.
            data["cases"][100]["binding"] = "different-output-hash"
            atomic(tmp_path / "review-data.json", data)
            page.locator("#refresh").click()
            page.wait_for_function("document.querySelector('#pick').value==='0'")
            page.locator("#pick").select_option("100")
            assert page.locator("[data-field=edit_fulfillment]").input_value() == ""
            assert (
                page.evaluate(
                    "JSON.parse(localStorage.getItem('qwen-unified-review-v1:test-only')).t100['binding-100'].preservation"
                )
                is True
            )
            assert not errors
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
