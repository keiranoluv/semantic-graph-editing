"""A synthetic 150-class graph exercises Toan's mixed review gate end to end."""

import csv
import hashlib
import json

import pytest
import torch

from scripts.apply_conceptnet_user_review import SELECTION_FIELDS, _csv_bytes, apply_review
from scripts.build_conceptnet_graph import write_outputs
from scripts.prepare_conceptnet_review import check, compile_reviewed, prepare
from scripts.suggest_conceptnet_review import FIELDS
from src.knowledge.complex import ComplexReviewed
from src.knowledge.conceptnet_builder import extract_assertions
from src.knowledge.export import export_conceptnet, load_class_features
from src.knowledge.rotate import RotateV2
from src.knowledge.transe.core import TransE, check_gate, load_kg, sha256


def test_mixed_direct_and_ai_review_is_bound_to_exact_decisions(tmp_path):
    mapping = {
        "schema_version": 1,
        "source": "wordnet",
        "rows": [{"class_id": 0, "label": "__background__", "status": "background"}]
        + [{"class_id": i, "label": f"item {i}", "status": "linked_auto"} for i in range(1, 151)],
    }
    snapshot = tmp_path / "assertions.tsv"
    snapshot.write_text(
        "".join(
            f"/a/{i}\t/r/IsA\t/c/en/item_{i}\t/c/en/object\t"
            '{"dataset":"/d/omcs","license":"cc:by-sa/4.0","weight":2.0}\n'
            for i in range(1, 151)
        )
    )
    kg_dir = tmp_path / "kg"
    outputs, _ = extract_assertions(
        mapping, snapshot, expected_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest()
    )
    write_outputs(kg_dir, outputs)
    source_dir = tmp_path / "source"
    outputs, _ = prepare(kg_dir)
    write_outputs(source_dir, outputs)
    originals = list(csv.DictReader((source_dir / "review.csv").open()))
    proposals = []
    for i, row in enumerate(originals, start=1):
        focus = i > 134
        proposals.append(
            {
                "class_id": i,
                "label": row["label"],
                "priority": row["priority"],
                "representative_wordnet_synset": "object.n.01",
                "wordnet_gloss": "an object",
                "conceptnet_uri_candidates": row["conceptnet_uri_candidates"],
                "proposed_decision": "defer" if focus and i < 149 else "reject" if focus else "accept",
                "proposed_uri": "" if focus else f"/c/en/item_{i}",
                "evidence_reference": f"/a/{i}",
                "evidence_summary": "out IsA /c/en/object",
                "ai_rationale": "Synthetic AI proposal for provenance testing.",
                "proposed_by": "Codex_AI_provisional",
                "requested_human_approver": "Toan",
            }
        )
    proposal_bytes = _csv_bytes(proposals, FIELDS)
    focus_bytes = _csv_bytes(proposals[134:], FIELDS)
    write_outputs(
        source_dir,
        {
            "review_suggestions.ai-draft.csv": proposal_bytes,
            "review_focus.ai-draft.csv": focus_bytes,
            "review_suggestions.ai-draft.json": json.dumps(
                {
                    "status": "AI_PROVISIONAL_NOT_HUMAN_REVIEWED_NOT_TRAINING_READY",
                    "requested_human_approver": "Toan",
                }
            ).encode(),
        },
    )
    selection_bytes = _csv_bytes(
        [
            {
                "class_id": i,
                "selected_uri": f"/c/en/item_{i}",
                "evidence_reference": f"/a/{i}",
                "rationale": "Toan accepts this synthetic class; URI chosen by Codex.",
            }
            for i in range(135, 139)
        ],
        SELECTION_FIELDS,
    )
    selection_file = tmp_path / "selection.csv"
    selection_file.write_bytes(selection_bytes)
    approval_file = tmp_path / "approval.json"
    approval = {
        "schema_version": 1,
        "approval_scope": "conceptnet_ai_134_plus_16_direct_focus",
        "approver": "Toan",
        "proposal_sha256": hashlib.sha256(proposal_bytes).hexdigest(),
        "focus_sha256": hashlib.sha256(focus_bytes).hexdigest(),
        "selection_sha256": hashlib.sha256(selection_bytes).hexdigest(),
        "accepted_focus_class_ids": [135, 136, 137, 138],
        "reject_remaining_focus": True,
        "approve_non_focus_ai_proposals": True,
        "statement": "I approve the other AI proposals and decide these focus IDs directly.",
        "approved_at_utc": "2026-10-25T00:00:00Z",
    }
    approval_file.write_text(json.dumps(approval))
    reviewed_outputs, provenance = apply_review(kg_dir, source_dir, approval_file, selection_file)
    assert provenance["review_basis"] == "human_mixed_direct_and_ai_batch"
    reviewed_dir = tmp_path / "reviewed"
    write_outputs(reviewed_dir, reviewed_outputs)
    assert check(kg_dir, reviewed_dir, require_complete=True) == {
        "accept": 138,
        "reject": 12,
        "defer": 0,
        "pending": 0,
    }
    compiled_dir = tmp_path / "compiled"
    write_outputs(compiled_dir, compile_reviewed(kg_dir, reviewed_dir, compiled_dir))
    kg = load_kg(compiled_dir)
    manifest = compiled_dir / "review_manifest.json"
    check_gate("reviewed", kg, compiled_dir, manifest)
    assert kg.metadata["human_verified_class_count"] == 16
    assert kg.metadata["human_batch_approved_class_count"] == 134

    vocabulary = tmp_path / "vocab.json"
    vocabulary.write_text(json.dumps({"label_to_idx": {f"item {i}": i for i in range(1, 151)}}))
    checkpoint = tmp_path / "best.pt"
    torch.save(
        {
            "manifest": {
                "milestone": "C.1",
                "mode": "reviewed",
                "kg_source": "conceptnet",
                "review_status": "human_approved",
                "review_basis": "human_mixed_direct_and_ai_batch",
                "review_manifest_sha256": sha256(manifest),
                "input_sha256": kg.hashes,
                "dim": 8,
            },
            "model": TransE(len(kg.entities), len(kg.relations), 8).state_dict(),
            "epoch": 1,
        },
        checkpoint,
    )
    features_dir = tmp_path / "features"
    export_conceptnet(checkpoint, compiled_dir, vocabulary, features_dir, review_manifest_path=manifest)
    _, valid, export_manifest = load_class_features(features_dir)
    assert int(valid.sum()) == 138
    assert valid[135:139].all() and not valid[139:].any()
    assert export_manifest["review_basis"] == "human_mixed_direct_and_ai_batch"

    rotate_checkpoint = tmp_path / "rotate-best.pt"
    rotate_manifest = {
        **torch.load(checkpoint, map_location="cpu", weights_only=True)["manifest"],
        "embedding_model": "rotate",
        "embedding_version": 2,
        "score_norm": "complex_l1",
        "loss": "filtered_self_adversarial_logsigmoid",
        "negative_count": 8,
        "complex_dim": 4,
        "entity_export": "real_imag_concat_l2_normalized",
    }
    torch.save(
        {
            "manifest": rotate_manifest,
            "model": RotateV2(len(kg.entities), len(kg.relations), 4).state_dict(),
            "epoch": 1,
        },
        rotate_checkpoint,
    )
    rotate_features_dir = tmp_path / "rotate-features"
    export_conceptnet(
        rotate_checkpoint,
        compiled_dir,
        vocabulary,
        rotate_features_dir,
        review_manifest_path=manifest,
        embedding_model="rotate",
    )
    rotate_features, rotate_valid, rotate_export_manifest = load_class_features(rotate_features_dir)
    assert int(rotate_valid.sum()) == 138
    assert rotate_features.shape == (151, 8)
    assert torch.allclose(torch.linalg.vector_norm(rotate_features[rotate_valid], dim=1), torch.ones(138), atol=1e-6)
    assert rotate_export_manifest["embedding_model"] == "rotate"
    rotate_state = torch.load(rotate_checkpoint, map_location="cpu", weights_only=True)["model"]
    first_entity_id = json.loads((compiled_dir / "class_entity_ids.json").read_text())["class_entity_ids"][1]
    expected = torch.nn.functional.normalize(
        torch.cat(
            (rotate_state["node_emb.weight"][first_entity_id], rotate_state["node_emb_im.weight"][first_entity_id])
        ),
        dim=0,
    )
    assert torch.allclose(rotate_features[1], expected)
    with pytest.raises(ValueError, match="embedding model"):
        export_conceptnet(
            rotate_checkpoint,
            compiled_dir,
            vocabulary,
            tmp_path / "wrong-model",
            review_manifest_path=manifest,
        )
    v1_checkpoint = tmp_path / "rotate-v1.pt"
    torch.save(
        {
            "manifest": {**rotate_manifest, "embedding_version": 1, "score_norm": "complex_l2"},
            "model": torch.load(rotate_checkpoint, map_location="cpu", weights_only=True)["model"],
            "epoch": 1,
        },
        v1_checkpoint,
    )
    with pytest.raises(ValueError, match="RotatE checkpoint"):
        export_conceptnet(
            v1_checkpoint,
            compiled_dir,
            vocabulary,
            tmp_path / "v1-output",
            review_manifest_path=manifest,
            embedding_model="rotate",
        )

    complex_checkpoint = tmp_path / "complex-best.pt"
    complex_manifest = {
        **torch.load(checkpoint, map_location="cpu", weights_only=True)["manifest"],
        "embedding_model": "complex",
        "embedding_version": 1,
        "score_function": "real_hermitian_trilinear_product",
        "loss": "filtered_logistic_l2",
        "negative_count": 4,
        "complex_dim": 4,
        "entity_export": "real_imag_concat_l2_normalized",
    }
    torch.save(
        {
            "manifest": complex_manifest,
            "model": ComplexReviewed(len(kg.entities), len(kg.relations), 4).state_dict(),
            "epoch": 1,
        },
        complex_checkpoint,
    )
    complex_features_dir = tmp_path / "complex-features"
    export_conceptnet(
        complex_checkpoint,
        compiled_dir,
        vocabulary,
        complex_features_dir,
        review_manifest_path=manifest,
        embedding_model="complex",
    )
    complex_features, complex_valid, complex_export_manifest = load_class_features(complex_features_dir)
    assert complex_features.shape == (151, 8)
    assert int(complex_valid.sum()) == 138
    assert torch.count_nonzero(complex_features[~complex_valid]) == 0
    assert torch.allclose(torch.linalg.vector_norm(complex_features[complex_valid], dim=1), torch.ones(138), atol=1e-6)
    assert complex_export_manifest["embedding_model"] == "complex"
    complex_state = torch.load(complex_checkpoint, map_location="cpu", weights_only=True)["model"]
    complex_expected = torch.nn.functional.normalize(
        torch.cat(
            (complex_state["node_emb.weight"][first_entity_id], complex_state["node_emb_im.weight"][first_entity_id])
        ),
        dim=0,
    )
    assert torch.allclose(complex_features[1], complex_expected)

    direct_file = compiled_dir / "direct_decisions.csv"
    direct = direct_file.read_bytes()
    direct_file.write_bytes(direct + b"\n")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        check_gate("reviewed", load_kg(compiled_dir), compiled_dir, manifest)
    direct_file.write_bytes(direct)
    approval["accepted_focus_class_ids"] = [135, 136, 137]
    approval_file.write_text(json.dumps(approval))
    with pytest.raises(ValueError, match="direct selection IDs"):
        apply_review(kg_dir, source_dir, approval_file, selection_file)
