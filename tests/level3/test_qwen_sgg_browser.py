"""Optional CPU Chromium test for the unified graph viewer, including pending and empty graphs."""

import functools
import http.server
import threading
from pathlib import Path

import pytest
from PIL import Image

from tools.level3_qwen_unified.core import atomic

pw = pytest.importorskip("playwright.sync_api")


def test_200_graph_pairs_refresh_boxes_confidences_and_safe_text(tmp_path):
    Image.new("RGB", (100, 80), "green").save(tmp_path / "image.jpg")
    graph = {
        "nodes": [{"id": "n000", "label": "person"}, {"id": "n001", "label": "chair"}],
        "edges": [{"id": "e000", "subject": "n000", "object": "n001", "predicate": "beside"}],
    }
    grounding = {
        "nodes": {
            "n000": {"bbox_xyxy": [0, 0, 30, 60], "confidence": 0.9},
            "n001": {"bbox_xyxy": [40, 0, 70, 60], "confidence": 0.8},
        },
        "edges": {"e000": {"confidence": 0.7, "triplet_confidence": 0.6}},
    }
    base = {
        "preview": "image.jpg",
        "native_image": "image.jpg",
        "raw_npz": "native.npz",
        "graph_json": "graph.json",
        "image": {"width": 100, "height": 80},
        "status": "ok",
        "graph": graph,
        "grounding": grounding,
    }
    cases = [
        {
            "task_id": f"t{i}",
            "image_id": f"i{i % 100}",
            "prompt": "<script>window.poison=true</script>",
            "roles": {
                "source": base,
                "target": base if i < 100 else {**base, "graph": None, "graph_json": None, "status": "not_run"},
            },
        }
        for i in range(200)
    ]
    data = {"cases": cases, "ready_pairs": 100, "unique_images": 300, "ready_images": 200}
    atomic(tmp_path / "review-data.json", data)
    source = Path(__file__).resolve().parents[2] / "tools/level3_qwen_sgg/viewer.html"
    (tmp_path / "comparison.html").write_text(source.read_text())
    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_path))
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with pw.sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--disable-gpu"])
            page = browser.new_page()
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(f"http://127.0.0.1:{server.server_port}/comparison.html")
            page.wait_for_function("document.querySelectorAll('#pick option').length===200")
            assert page.locator("svg.scene").count() == 2
            assert page.locator("svg.graphs").count() == 2
            assert page.locator(".box").count() == 4
            assert page.locator("line[marker-end]").count() == 2
            assert "0.700 / 0.600" in page.locator("#pair").inner_text()
            assert page.evaluate("window.poison") is None
            assert page.locator('a:has-text("EditGraph JSON")').count() == 2
            page.locator("#boxes").click()
            assert page.locator(".scene.hide-boxes").count() == 2
            page.locator("#pick").select_option("100")
            assert page.locator("svg.graphs").count() == 1
            assert "Graph chưa có" in page.locator("#pair").inner_text()
            data["cases"][100]["roles"]["target"] = {
                **base,
                "graph": {"nodes": [], "edges": []},
                "grounding": {"nodes": {}, "edges": {}},
            }
            data.update(ready_pairs=101, ready_images=201)
            atomic(tmp_path / "review-data.json", data)
            page.locator("#refresh").click()
            page.wait_for_function("document.querySelector('#count').textContent.includes('101/200')")
            assert page.locator("#pick").input_value() == "100"
            assert "Không có node được dự đoán" in page.locator("#pair").inner_text()
            assert "Graph chưa có" not in page.locator("#pair").inner_text()
            assert not errors
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
