"""Guard the frozen split and input/target boundary without loading real labels."""

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest
from jsonschema import ValidationError

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("pvsg_m0_validator", ROOT / "validate_m0.py")
m0 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m0)


@pytest.fixture
def manifest():
    return m0.read_json(ROOT / "manifests" / "pvsg_author_release_338_62.lock.json")


def test_selected_split_and_synthetic_contracts(manifest):
    m0.validate_manifest(manifest, require_selected=True)
    assert len(manifest["outer_split"]["train_ids"]) == 338
    assert len(manifest["outer_split"]["heldout_ids"]) == 62
    assert manifest["internal_validation"]["state"] == "pending"
    assert manifest["ready_for_training"] is False
    m0.validate_examples()


def test_unresolved_paper_split_cannot_be_selected():
    template = m0.read_json(ROOT / "manifests" / "cvf_360_40.unresolved.template.json")
    m0.validate_manifest(template)
    with pytest.raises(ValueError, match="unresolved template"):
        m0.validate_manifest(template, require_selected=True)


def test_verified_candidate_does_not_imply_selection(manifest):
    manifest.update(state="verified_candidate", selected_by_user=False)
    m0.validate_manifest(manifest)
    with pytest.raises(ValueError, match="not been selected"):
        m0.validate_manifest(manifest, require_selected=True)


def test_outer_overlap_is_rejected_even_when_counts_unchanged(manifest):
    manifest["outer_split"]["heldout_ids"][0] = manifest["outer_split"]["train_ids"][0]
    with pytest.raises(ValueError, match="overlap"):
        m0.validate_manifest(manifest)


def test_wrong_count_is_rejected(manifest):
    manifest["outer_split"]["expected_counts"]["train"] = 360
    with pytest.raises(ValueError, match="counts"):
        m0.validate_manifest(manifest)


def test_disjoint_id_swap_is_detected_by_split_hash(manifest):
    outer = manifest["outer_split"]
    outer["train_ids"][0], outer["heldout_ids"][0] = outer["heldout_ids"][0], outer["train_ids"][0]
    with pytest.raises(ValueError, match="hash mismatch"):
        m0.validate_manifest(manifest)


def test_missing_annotation_identity_cannot_be_locked(manifest):
    manifest["annotation"]["sha256"] = None
    with pytest.raises(ValidationError):
        m0.validate_manifest(manifest, require_selected=True)


def test_internal_validation_can_only_partition_official_train(manifest):
    train = manifest["outer_split"]["train_ids"]
    internal = manifest["internal_validation"]
    internal.update(state="locked", fit_ids=train[1:], validation_ids=train[:1])
    m0.validate_manifest(manifest)
    internal["validation_ids"] = manifest["outer_split"]["heldout_ids"][:1]
    with pytest.raises(ValueError, match="official train only"):
        m0.validate_manifest(manifest)


@pytest.mark.parametrize("leaked_key", ["relations", "captions", "qa_pairs", "summary"])
def test_model_input_rejects_target_or_text_leakage(leaked_key):
    video = m0.read_json(ROOT / "examples" / "synthetic_video.json")
    video[leaked_key] = []
    with pytest.raises(ValidationError):
        m0.validate_document(video, "video.schema.json")


@pytest.mark.parametrize(
    "field,value", [("predicate_id", 57), ("score", 1.1), ("score", -0.1), ("intervals", [[-1, 3]])]
)
def test_prediction_rejects_invalid_ranges(field, value):
    predictions = m0.read_json(ROOT / "examples" / "synthetic_predictions.json")
    predictions["predictions"][0][field] = value
    with pytest.raises(ValidationError):
        m0.validate_document(predictions, "predictions.schema.json")


def test_annotation_byte_identity_precedes_parsing(tmp_path, manifest):
    changed = tmp_path / "changed_annotation.json"
    changed.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="size/hash mismatch"):
        m0.validate_annotation_identity(changed, manifest, {})


def test_vocab_order_is_part_of_annotation_identity(tmp_path):
    annotation = {
        "split": {"vidor": {"train": ["a"], "val": ["b"]}},
        "data": [{"video_id": vid, "meta": {"fps": 5}} for vid in ("a", "b")],
        "objects": {"thing": ["person", "cup"], "stuff": []},
        "relations": ["holding", "beside"],
    }
    path = tmp_path / "synthetic_annotation.json"
    path.write_text(json.dumps(annotation), encoding="utf-8")
    digest = m0.file_sha256(path)
    manifest = {
        "annotation": {"bytes": path.stat().st_size, "sha256": digest},
        "outer_split": {"train_ids": ["a"], "heldout_ids": ["b"]},
        "video_sources": {"a": "vidor", "b": "vidor"},
    }
    vocab = {"annotation_sha256": digest, "object_classes": ["person", "cup"], "predicates": ["holding", "beside"]}
    assert m0.validate_annotation_identity(path, manifest, vocab) == {"videos": 2, "train": 1, "heldout": 1}
    reordered = deepcopy(vocab)
    reordered["predicates"].reverse()
    with pytest.raises(ValueError, match="vocabulary order"):
        m0.validate_annotation_identity(path, manifest, reordered)
