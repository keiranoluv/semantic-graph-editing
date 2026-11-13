"""Small public metadata/HEAD checks only; never download model weights."""

import hashlib

import requests


def verify_model(cfg):
    model = cfg["model"]
    repo, revision = model["id"], model["revision"]
    session = requests.Session()

    def get(url):
        response = session.get(url, timeout=(10, 40))
        response.raise_for_status()
        if len(response.content) > 2 * 1024 * 1024:
            raise ValueError("model_metadata_size_limit")
        return response

    info = get(f"https://huggingface.co/api/models/{repo}/revision/{revision}").json()
    config = get(f"https://huggingface.co/{repo}/raw/{revision}/config.json")
    processor = get(f"https://huggingface.co/{repo}/raw/{revision}/processor_config.json")
    card = get(f"https://huggingface.co/{repo}/raw/{revision}/README.md")
    tree = get(f"https://huggingface.co/api/models/{repo}/tree/{revision}").json()
    parsed = config.json()
    if info["sha"] != revision or info.get("private") or info.get("gated"):
        raise ValueError("checkpoint_not_public_at_pinned_revision")
    if info.get("pipeline_tag") != "image-text-to-text" or not parsed.get("vision_config"):
        raise ValueError("checkpoint_image_support_unverified")
    if (
        model["architecture"] not in parsed["architectures"]
        or parsed["quantization_config"]["quant_method"] != model["quantization"]
    ):
        raise ValueError("checkpoint_architecture_or_quantization_changed")
    if hashlib.sha256(config.content).hexdigest() != model["config_sha256"]:
        raise ValueError("checkpoint_config_hash_mismatch")
    if info.get("cardData", {}).get("license") != model["license"]:
        raise ValueError("checkpoint_license_mismatch")
    weights = [r for r in tree if r["path"].endswith(".safetensors")]
    if not weights:
        raise ValueError("checkpoint_weights_missing")
    availability = {}
    for weight in weights:
        response = session.head(
            f"https://huggingface.co/{repo}/resolve/{revision}/{weight['path']}", allow_redirects=True, timeout=(10, 40)
        )
        response.raise_for_status()
        availability[weight["path"]] = {"bytes": weight["size"], "http_status": response.status_code}
    return {
        "model_id": repo,
        "revision": revision,
        "license": model["license"],
        "architecture": model["architecture"],
        "quantization": model["quantization"],
        "metadata_sha256": {
            "config.json": hashlib.sha256(config.content).hexdigest(),
            "processor_config.json": hashlib.sha256(processor.content).hexdigest(),
            "README.md": hashlib.sha256(card.content).hexdigest(),
        },
        "weights_accessible_via_HEAD_only": availability,
        "trust_remote_code": False,
        "model_weights_downloaded": False,
        "image_architecture_and_processor_present": True,
        "two_image_runtime_smoke_verified": False,
        "RTX5090_fit_measured_locally": False,
        "vllm_version_required": cfg["server"]["vllm_version"],
        "interpretation": "Public metadata and source-level support do not prove GPU kernel compatibility or visual annotation correctness.",
    }
