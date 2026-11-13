"""Keep the slow image transform, explicitly use Fast tokenizer for offsets."""

from tools.magicbrush_grounded_v1.models import decode_detections, query_text, release


def detector_processor(config):
    from transformers import AutoImageProcessor, AutoTokenizer, GroundingDinoProcessor

    image_processor = AutoImageProcessor.from_pretrained(
        config["detector_model"], revision=config["detector_revision"], use_fast=False
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config["detector_model"], revision=config["detector_revision"], use_fast=True
    )
    if not tokenizer.is_fast:
        raise ValueError("DINO requires a Fast tokenizer for exact query token offsets")
    return GroundingDinoProcessor(image_processor=image_processor, tokenizer=tokenizer)


def detector_inputs(processor, images, vocabularies):
    prompts = [query_text(queries) for queries in vocabularies]
    inputs = processor(
        images=images,
        text=[p[0] for p in prompts],
        padding=True,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    # Offsets come from exactly the padded encoding supplied to the model.
    # They are metadata, not an argument to GroundingDinoForObjectDetection.
    offsets = inputs.pop("offset_mapping").tolist()
    if inputs["input_ids"].shape[1] > 256:
        raise ValueError("DINO query batch exceeds the frozen 256-token budget")
    for mapping, (_, spans) in zip(offsets, prompts, strict=True):
        for start, end in spans:
            tokens = [(a, b) for a, b in mapping if b > a and a < end and b > start]
            if not tokens or min(a for a, _ in tokens) != start or max(b for _, b in tokens) != end:
                raise ValueError("DINO query token offsets do not cover the exact input span")
    return inputs, offsets, prompts


def detect_images(images, vocabularies, foci, config):
    import torch
    from transformers import AutoModelForZeroShotObjectDetection

    processor = detector_processor(config)
    # Validate all actual query batches BEFORE loading/launching the GPU model.
    batch_size = config["batch_size"]
    for start in range(0, len(images), batch_size):
        detector_inputs(processor, images[start : start + batch_size], vocabularies[start : start + batch_size])
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        config["detector_model"],
        revision=config["detector_revision"],
        disable_custom_kernels=config["detector_disable_custom_kernels"],
    )
    model.to("cuda").eval().requires_grad_(False)
    result = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            vocab = vocabularies[start : start + batch_size]
            inputs, offsets, prompts = detector_inputs(processor, images[start : start + batch_size], vocab)
            outputs = model(**inputs.to("cuda"))
            for k, (_, spans) in enumerate(prompts):
                result.append(
                    decode_detections(
                        outputs.logits[k].float().cpu().numpy(),
                        outputs.pred_boxes[k].float().cpu().numpy(),
                        offsets[k],
                        spans,
                        vocab[k],
                        foci[start + k],
                        config,
                    )
                )
    model.to("cpu")
    release(model)
    return result
