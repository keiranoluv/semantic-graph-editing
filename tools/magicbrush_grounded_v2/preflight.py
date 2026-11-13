"""Validate every real pilot query on CPU; no model weights or CUDA compute."""

import json

from src.knowledge.transe.core import sha256
from tools.magicbrush_grounded_v1.language import caption_nouns, mentions, scope
from tools.magicbrush_grounded_v2.captions import verified_captions
from tools.magicbrush_grounded_v2.models import detector_inputs, detector_processor
from tools.prepare_magicbrush import encode, fixed


def preflight(config, output):
    import torch

    from tools.magicbrush_grounded_v2.runner import load_image

    bundle = verified_captions(config, output)
    if bundle is None:
        raise ValueError("No parent captions; run this preflight after source caption inference")
    captions = {r["id"]: r["caption"] for r in bundle["entries"]}
    manifest = json.loads((output / "manifest.json").read_text())
    rows = [r for r in manifest["records"] if r["id"] in manifest["review_ids"]]
    if len(rows) != 20 or any(r["split"] != "train" or r["official_split"] != "train" for r in rows):
        raise ValueError("Preflight requires exactly the frozen20 train pairs")
    jobs = []
    for row in rows:
        caption = captions[row["source_image"]["sha256"]]
        focus = mentions(scope(row["instruction"]))
        queries = (focus + [q for q in caption_nouns(caption) if q not in focus])[: config["max_queries"]]
        for role in ("source", "target"):
            jobs.append((row, role, queries))
    processor = detector_processor(config)
    slow = processor.tokenizer.slow_tokenizer_class.from_pretrained(
        config["detector_model"], revision=config["detector_revision"]
    )
    token_counts = []
    for start in range(0, len(jobs), config["batch_size"]):
        batch = jobs[start : start + config["batch_size"]]
        inputs, _, prompts = detector_inputs(
            processor, [load_image(r[f"{role}_image"]) for r, role, _ in batch], [q for _, _, q in batch]
        )
        for i, (text, _) in enumerate(prompts):
            ids = inputs["input_ids"][i][inputs["attention_mask"][i].bool()].tolist()
            if ids != slow(text)["input_ids"]:
                raise ValueError("Fast/slow token IDs differ for an actual pilot query")
            token_counts.append(len(ids))
    report = {
        "schema_version": 1,
        "GPU_started": False,
        "CUDA_initialized": torch.cuda.is_initialized(),
        "pretrained_model_inference": False,
        "train_pairs": len(rows),
        "image_query_jobs": len(jobs),
        "reusable_captions": len(captions),
        "tokenizer": type(processor.tokenizer).__name__,
        "image_processor": type(processor.image_processor).__name__,
        "same_batch_offsets_checked": True,
        "fast_slow_token_ids_equal_for_all_queries": True,
        "max_query_tokens": max(token_counts),
        "parent_identity_sha256": bundle["parent_identity_sha256"],
        "manifest_sha256": sha256(output / "manifest.json"),
    }
    fixed(output / "cpu-preflight.json", encode(report))
    return report
