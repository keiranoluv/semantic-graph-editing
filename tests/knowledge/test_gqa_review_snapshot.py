"""Confirmed reviews load once without rewriting evidence or later browser edits."""

import copy
import json

import pytest

from src.knowledge.transe.core import sha256
from tests.knowledge import test_gqa_human_review as fixtures
from tests.knowledge.test_gqa_human_review_browser import browser
from tools import apply_gqa_review_snapshot as snapshot
from tools import prepare_gqa_wordnet_review as review

local_review_server = fixtures.local_review_server
packet = fixtures.packet
browser = browser


@pytest.fixture
def confirmed(local_review_server, monkeypatch):
    url, output, packet = local_review_server
    monkeypatch.setattr(snapshot, "ROOT", review.ROOT)
    original_hash = packet["packet_sha256"]
    packet["packet_sha256"] = review.canonical_hash({k: v for k, v in packet.items() if k != "packet_sha256"})
    index = output / "index.html"
    index.write_text(index.read_text().replace(original_hash, packet["packet_sha256"]))
    (output / "review-data.json").write_text(json.dumps(packet))
    (output / "verification.json").write_text(
        json.dumps({"files": {str(index.relative_to(review.ROOT)): sha256(index)}})
    )
    choices = {
        "20": {
            "decision": "accept",
            "selected_synset": "fixture.n.01",
            "reviewer": "Fixture QA",
            "notes": "Approved fixture",
        },
        "604": {"decision": "defer", "selected_synset": "", "reviewer": "Fixture QA", "notes": ""},
    }
    source = output / "confirmed.csv"
    source.write_bytes(review.export_csv(packet, choices))
    return url, output, source, packet, choices


def test_snapshot_is_idempotent_and_preserves_packet_and_original_html(confirmed):
    _, output, source, _, _ = confirmed
    before = {
        name: (output / name).read_bytes() for name in ("index.html", "review-data.json", "viewer.js", "confirmed.csv")
    }
    first = snapshot.apply_snapshot(output, source)
    index = (output / "index.html").read_bytes()
    assert snapshot.apply_snapshot(output, source) == first
    assert (output / "index.html").read_bytes() == index
    assert (output / "index-before-confirmed-review.html").read_bytes() == before["index.html"]
    for name in ("review-data.json", "viewer.js", "confirmed.csv"):
        assert (output / name).read_bytes() == before[name]
    assert first["decision_counts"] == {"accept": 1, "defer": 1}


def test_snapshot_rejects_pending_rows(confirmed):
    _, output, source, packet, choices = confirmed
    choices = copy.deepcopy(choices)
    choices["604"]["decision"] = ""
    source.write_bytes(review.export_csv(packet, choices))
    before = (output / "index.html").read_bytes()
    with pytest.raises(ValueError, match="explicit decision"):
        snapshot.apply_snapshot(output, source)
    assert (output / "index.html").read_bytes() == before


def test_browser_imports_confirmed_snapshot_backs_up_draft_and_retains_later_edits(browser, confirmed):
    url, output, source, packet, choices = confirmed
    snapshot.apply_snapshot(output, source)
    context = browser.new_context()
    page = context.new_page()
    page.goto(url)
    page.wait_for_function("document.getElementById('progress').textContent.startsWith('2/2')")
    key = f"gqa-wordnet-review:{packet['packet_sha256']}"
    old = copy.deepcopy(choices)
    old["604"]["decision"] = ""
    draft = {"reviews": old, "current": "604", "defaultReviewer": "Fixture QA"}
    page.evaluate(
        "([key, draft]) => {localStorage.clear();localStorage.setItem(key,JSON.stringify(draft));}", [key, draft]
    )
    page.reload()
    page.wait_for_function("document.getElementById('progress').textContent.startsWith('2/2')")
    assert page.locator("#decision").input_value() == "defer"
    backup_key = f"{key}:before-snapshot:{sha256(source)}"
    assert page.evaluate("key => JSON.parse(localStorage.getItem(key))", backup_key) == draft
    page.locator("#notes").fill("Later human edit")
    page.reload()
    assert page.locator("#notes").input_value() == "Later human edit"
    assert page.locator("#decision").input_value() == "defer"
    file_page = context.new_page()
    file_page.goto((output / "index.html").as_uri())
    file_page.wait_for_function("document.getElementById('progress').textContent.startsWith('2/2')")
    assert file_page.locator("#save-csv").is_disabled()
    context.close()
