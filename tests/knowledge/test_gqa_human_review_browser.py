"""Browser QA uses synthetic evidence only; never creates real review decisions."""

import csv
import io

import pytest

from tests.knowledge import test_gqa_human_review as fixtures

local_review_server = fixtures.local_review_server
packet = fixtures.packet

playwright = pytest.importorskip("playwright.sync_api")


@pytest.fixture
def browser():
    with playwright.sync_playwright() as runtime:
        try:
            browser = runtime.chromium.launch()
        except playwright.Error as error:
            pytest.skip(f"Optional review browser environment unavailable: {error}")
        yield browser
        browser.close()


def test_review_choices_csv_export_import_persistence_and_explicit_save(browser, local_review_server, tmp_path):
    url, output, _ = local_review_server
    context = browser.new_context(accept_downloads=True)
    page = context.new_page()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(url)
    assert page.locator("#progress").inner_text().startswith("0/2")
    assert page.locator("#decision").input_value() == ""
    assert not page.locator("input[name=synset]").is_checked()
    page.locator("input[name=synset]").check()
    assert page.locator("#decision").input_value() == ""  # Selection alone is not approval.
    page.locator("#decision").select_option("accept")
    page.locator("#export-csv").click()
    assert "thiếu tên" in page.locator("#message").inner_text()
    page.locator("#reviewer").fill("Fixture QA")
    notes = 'Định nghĩa "mẫu", dòng 1\ndòng 2'
    page.locator("#notes").fill(notes)
    with page.expect_download() as download:
        page.locator("#export-csv").click()
    exported = tmp_path / "synthetic-review.csv"
    download.value.save_as(exported)
    text = exported.read_text(encoding="utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text)))
    assert rows[0]["class_id"] == "20" and rows[0]["label"] == "fixture animal"
    assert rows[0]["notes"] == notes and rows[0]["decision"] == "accept"
    assert rows[1]["decision"] == ""
    page.reload()
    assert page.locator("#notes").input_value() == notes
    page.locator('[data-class-id="604"]').click()
    assert page.locator("#decision option[value=accept]").is_disabled()
    assert "Chưa có ứng viên" in page.locator("#class-warning").inner_text()
    page.locator("#default-reviewer").fill("Fixture QA")
    page.locator("#decision").select_option("defer")
    page.locator("#save-csv").click()
    page.wait_for_function("document.getElementById('message').textContent.startsWith('Đã lưu')")
    files = list((output / "submissions").glob("*.csv"))
    assert len(files) == 1
    text = files[0].read_text(encoding="utf-8-sig")
    saved = list(csv.DictReader(io.StringIO(text)))
    assert [r["decision"] for r in saved] == ["accept", "defer"]
    page.evaluate("localStorage.clear()")
    page.reload()
    page.locator("#import-csv").set_input_files(files[0])
    assert "Đã nhập CSV" in page.locator("#message").inner_text()
    assert page.locator("#progress").inner_text().startswith("2/2")
    page.locator("#import-csv").set_input_files(
        {
            "name": "bad.csv",
            "mimeType": "text/csv",
            "buffer": files[0].read_bytes().replace(b"fixture animal", b"wrong label"),
        }
    )
    assert "khác class ID" in page.locator("#message").inner_text()
    assert page.locator("#progress").inner_text().startswith("2/2")
    assert errors == []
    context.close()


def test_file_mode_mobile_layout_and_image_loading(browser, local_review_server):
    _, output, _ = local_review_server
    context = browser.new_context(viewport={"width": 390, "height": 844})
    page = context.new_page()
    page.goto((output / "index.html").as_uri())
    assert page.locator("#save-csv").is_disabled()
    page.wait_for_function(
        "[...document.querySelectorAll('.examples img')].every(i => i.complete && i.naturalWidth > 0)"
    )
    assert page.locator(".examples img").count() == 2
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    assert page.locator("#decision").input_value() == ""
    context.close()
