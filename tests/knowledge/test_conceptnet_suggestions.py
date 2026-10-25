"""AI proposal test uses invented assertions; it never asserts human approval."""

import csv
import hashlib
import io
import json
from types import SimpleNamespace

from test_conceptnet_review import _fixture

from scripts import suggest_conceptnet_review
from scripts.approve_conceptnet_suggestions import apply_approval
from scripts.build_conceptnet_graph import write_outputs
from scripts.prepare_conceptnet_review import check, compile_reviewed
from src.knowledge.transe.core import check_gate, load_kg


def test_ai_suggestions_do_not_fill_human_review(tmp_path, monkeypatch):
    kg_dir, review_dir = _fixture(tmp_path)
    with (review_dir / "review.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    mapping_path = tmp_path / "representatives.json"
    mapping_path.write_text(
        json.dumps(
            {
                "rows": [{"class_id": 0, "label": "__background__"}]
                + [{"class_id": i, "label": row["label"], "synset_id": "dog.n.01"} for i, row in enumerate(rows, 1)]
            }
        )
    )
    fake_synset = SimpleNamespace(definition=lambda: "a domestic animal", hypernyms=lambda: [])
    monkeypatch.setattr(suggest_conceptnet_review.wordnet, "synset", lambda _: fake_synset)
    monkeypatch.setattr(
        suggest_conceptnet_review, "EXPECTED_MAPPING_SHA256", suggest_conceptnet_review._sha256(mapping_path)
    )
    outputs, provenance = suggest_conceptnet_review.build(kg_dir, review_dir, mapping_path, approver="Toan")
    suggestions = list(csv.DictReader(io.StringIO(outputs["review_suggestions.ai-draft.csv"].decode())))
    focus = list(csv.DictReader(io.StringIO(outputs["review_focus.ai-draft.csv"].decode())))
    assert len(suggestions) == 150
    assert suggestions[0]["proposed_decision"] == "accept"
    assert suggestions[0]["proposed_uri"] == "/c/en/dog"
    assert suggestions[0]["evidence_reference"] == "/a/dog"
    assert suggestions[10]["proposed_decision"] == "defer"
    assert "reviewer" not in suggestions[0]
    assert suggestions[0]["requested_human_approver"] == "Toan"
    assert provenance["status"] == "AI_PROVISIONAL_NOT_HUMAN_REVIEWED_NOT_TRAINING_READY"
    assert len(focus) == provenance["human_focus_count"]
    assert all(row["proposed_decision"] != "accept" for row in focus)
    assert (review_dir / "review.csv").read_text().count(",accept,") == 0

    write_outputs(review_dir, outputs)
    approval = tmp_path / "approval.json"
    approval.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "approval_scope": "all_150_conceptnet_ai_proposals",
                "approver": "Toan",
                "proposal_sha256": hashlib.sha256(outputs["review_suggestions.ai-draft.csv"]).hexdigest(),
                "focus_sha256": hashlib.sha256(outputs["review_focus.ai-draft.csv"]).hexdigest(),
                "statement": "I approve this synthetic AI proposal as a batch for testing only.",
                "approved_at_utc": "2026-10-25T00:00:00Z",
            }
        )
    )
    approved_outputs, approved_provenance = apply_approval(kg_dir, review_dir, approval)
    assert approved_provenance["review_basis"] == "human_batch_approved_ai_proposal"
    approved_dir = tmp_path / "approved"
    write_outputs(approved_dir, approved_outputs)
    assert check(kg_dir, approved_dir, require_complete=True)["pending"] == 0
    reviewed_dir = tmp_path / "reviewed"
    write_outputs(reviewed_dir, compile_reviewed(kg_dir, approved_dir, reviewed_dir))
    kg = load_kg(reviewed_dir)
    assert kg.metadata["human_verified_class_count"] == 0
    assert kg.metadata["human_batch_approved_class_count"] == 150
    check_gate("reviewed", kg, reviewed_dir, reviewed_dir / "review_manifest.json")
