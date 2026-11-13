"""Regression for the actual slow-image/Fast-tokenizer production factory."""

import json

import numpy as np
import pytest
import torch
from PIL import Image

from tools.magicbrush_grounded_v1.models import decode_detections
from tools.magicbrush_grounded_v2 import captions
from tools.magicbrush_grounded_v2.models import detector_inputs, detector_processor
from tools.prepare_magicbrush import ROOT, digest, encode

CONFIG = json.loads((ROOT / "configs/level2/magicbrush-grounded-anchor-v2.json").read_text())


@pytest.fixture(scope="module")
def processor():
    return detector_processor(CONFIG)


def test_actual_processor_factory_offsets_and_batch_padding(processor):
    assert processor.tokenizer.is_fast is True
    assert type(processor.image_processor).__name__ == "GroundingDinoImageProcessor"
    queries = [["duck", "fire hydrant"], ["pizza cutter"]]
    inputs, offsets, prompts = detector_inputs(
        processor, [Image.new("RGB", (500, 500)), Image.new("RGB", (1024, 1024))], queries
    )
    assert "offset_mapping" not in inputs
    assert inputs["pixel_values"].shape == (2, 3, 800, 800)
    assert len(offsets[0]) == len(offsets[1]) == inputs["input_ids"].shape[1]
    assert inputs["attention_mask"][1, -1] == 0  # different lengths, real padding
    slow = processor.tokenizer.slow_tokenizer_class.from_pretrained(
        CONFIG["detector_model"], revision=CONFIG["detector_revision"]
    )
    for i, (text, spans) in enumerate(prompts):
        token_ids = inputs["input_ids"][i][inputs["attention_mask"][i].bool()].tolist()
        assert token_ids == slow(text)["input_ids"]
        logits = np.full((1, 256), -np.inf)
        logits[0, : len(token_ids)] = 3.0
        prediction = decode_detections(
            logits, np.array([[0.5, 0.5, 0.2, 0.2]]), offsets[i], spans, queries[i], queries[i], CONFIG
        )
        assert set(prediction["presence_scores"]) == set(queries[i])
        assert all(s > 0.9 for s in prediction["presence_scores"].values())


def test_offsets_supported_in_production_path_instead_of_v1_slow_error(processor):
    slow = processor.tokenizer.slow_tokenizer_class.from_pretrained(
        CONFIG["detector_model"], revision=CONFIG["detector_revision"]
    )
    with pytest.raises(NotImplementedError):
        slow("duck. fire hydrant.", return_offsets_mapping=True)
    encoded = processor.tokenizer("duck. fire hydrant.", return_offsets_mapping=True)
    assert encoded["offset_mapping"][1] == (0, 4)


def test_caption_import_preserves_parent_and_checks_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(captions, "ROOT", tmp_path)
    source = tmp_path / "data/processed/level2/parent"
    output = tmp_path / "data/processed/level2/fixed"
    source.mkdir(parents=True)
    output.mkdir(parents=True)
    config = {**CONFIG, "caption_import_source": str(source.relative_to(tmp_path))}
    manifest = {
        "records": [{"id": "train-fixture", "source_image": {"sha256": "image-fixture"}}],
        "splits": {"train": ["train-fixture"]},
        "review_ids": ["train-fixture"],
    }
    for directory in (source, output):
        (directory / "manifest.json").write_bytes(encode(manifest))
    identity = {
        "source_sha256": {},
        "config": CONFIG,
        "dataset_manifest_sha256": captions.sha256(source / "manifest.json"),
    }
    (source / "inference-identity.json").write_bytes(encode(identity))
    path = source / "captions/image-fixture.json"
    path.parent.mkdir()
    content = encode({"id": "image-fixture", "identity_sha256": digest(identity), "caption": "unit fixture only"})
    path.write_bytes(content)
    assert captions.import_captions(config, output, "new-fixture-fingerprint") == 1
    imported = json.loads((output / "captions/image-fixture.json").read_text())
    assert imported["caption"] == "unit fixture only"
    assert imported["identity_sha256"] == "new-fixture-fingerprint"
    assert imported["imported_from"]["sha256"] == captions.sha256(path)
    assert path.read_bytes() == content
    assert captions.import_captions(config, output, "new-fixture-fingerprint") == 1
    broken = json.loads(path.read_text())
    broken["identity_sha256"] = "wrong-parent"
    path.write_bytes(encode(broken))
    with pytest.raises(ValueError, match="Parent caption identity"):
        captions.verified_captions(config, output)
    path.write_bytes(content)
    with pytest.raises(ValueError, match="model/recipe differs"):
        captions.verified_captions({**config, "caption_revision": "different"}, output)


def test_oversized_query_fails_on_cpu_before_model_loading(processor):
    with pytest.raises(ValueError, match="256-token budget"):
        detector_inputs(processor, [Image.new("RGB", (32, 32))], [["duck " * 260]])


def test_cuda_was_not_initialized_by_cpu_tests():
    assert not torch.cuda.is_initialized()
