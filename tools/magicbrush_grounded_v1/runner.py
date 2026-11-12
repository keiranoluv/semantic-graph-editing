"""CPU preparation/compilation and a bounded, USER-RUN frozen GPU pilot."""

import argparse
import base64
import hashlib
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from src.datasets.gqa_full import atomic_write
from src.knowledge.transe.core import sha256
from tools.magicbrush_grounded_v1.language import TAGGER, TAGGER_FILES, caption_nouns, mentions, scope
from tools.magicbrush_grounded_v1.policy import analyze
from tools.prepare_magicbrush import DATA, ROOT, digest, encode, fixed
from tools.run_gqa_full import exclusive

CONFIG = ROOT / "configs/level2/magicbrush-grounded-anchor-v1.json"
BASE_SHA = "4e815988bb39c130fb912ce5aea0686317d6a0c1647840cbc204b6317834f776"


def settings(path):
    config = json.loads(path.read_text())
    if (
        config["device"] != "cuda"
        or config["dtype"] != "float32"
        or type(config["batch_size"]) is not int
        or not 1 <= config["batch_size"] <= 8
    ):
        raise ValueError("Recipe requires CUDA float32, detector/caption batch 1..8")
    if config["pilot_pairs"] != 20 or not 1 <= config["max_nodes"] <= 6 or not 2 <= config["max_queries"] <= 10:
        raise ValueError("This recipe is a fixed 20-pair, bounded pilot")
    output = (ROOT / config["output"]).resolve()
    if not output.is_relative_to(ROOT / "data/processed/level2"):
        raise ValueError("Use a new data/processed/level2/ output directory")
    return config, output


def prepare_tagger():
    import requests

    from tools.magicbrush_grounded_v1.language import tagger

    paths = [
        TAGGER / f"taggers/averaged_perceptron_tagger_eng/averaged_perceptron_tagger_eng.{suffix}.json"
        for suffix in TAGGER_FILES
    ]
    if not all(p.exists() for p in paths):
        url = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/taggers/averaged_perceptron_tagger_eng.zip"
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        if (
            hashlib.sha256(response.content).hexdigest()
            != "6025f530624335c67d6547d44757b357b4e79bae030a0383e9887a92c1718f0b"
        ):
            raise ValueError("POS tagger download changed")
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            for path in paths:
                # Explicit three-file whitelist; no arbitrary archive extraction.
                relative = path.relative_to(TAGGER).as_posix()
                fixed(path, archive.read(relative.removeprefix("taggers/")))
    tagger()


def prepare(config, output, base_path):
    prepare_tagger()
    if sha256(base_path) != BASE_SHA:
        raise ValueError("Expected the frozen 602-pair MagicBrush v1 manifest")
    base = json.loads(base_path.read_text())
    train = [r for r in base["records"] if r["split"] == "train"]
    if any(r["official_split"] != "train" for r in base["records"]):
        raise ValueError("Official dev/test data is forbidden")
    rows = [{"id": r["id"], "scope": scope(r["instruction"])} for r in train]
    supported = [r["id"] for r in rows if r["scope"]["supported"]]
    selected = sorted(supported, key=lambda identifier: (digest([config["seed"], identifier]), identifier))[:20]
    if len(selected) != 20:
        raise ValueError("Fewer than 20 in-scope train requests; do not start GPU")
    manifest = {
        **base,
        "review_ids": selected,
        "original_review_ids": base["review_ids"],
        "base_manifest_sha256": sha256(base_path),
        "selection": "20 grammar-supported train IDs by hash(seed,id), selected before any model outputs; no replacement after failures",
        "annotation_method": "grounded_source_anchor_spatial_v1",
        "graph_edit_requirement_complete": False,
    }
    fixed(output / "manifest.json", encode(manifest))
    plan = {
        "schema_version": 1,
        "GPU_started": False,
        "base_manifest_sha256": sha256(base_path),
        "raw_pairs_preserved": len(base["records"]),
        "train_pairs_scoped": len(train),
        "grammar_supported_train": len(supported),
        "scope_counts": dict(Counter(r["scope"].get("kind", r["scope"].get("reason")) for r in rows)),
        "pilot_ids": selected,
        "scope_records": rows,
        "scope_is_not_label_quality": True,
        "val_test_semantic_decisions": False,
        "minimum_300_valid_labels_met": False,
        "ready_for_training": False,
    }
    fixed(output / "preparation.json", encode(plan))
    return plan


def identity(config, output):
    files = sorted((ROOT / "tools/magicbrush_grounded_v1").glob("*.py")) + [
        ROOT / "tools/annotate_magicbrush_grounded_v1.py",
        ROOT / "tools/prepare_magicbrush.py",
        ROOT / "tools/magicbrush_graphs.py",
        ROOT / "src/editing/operations.py",
        ROOT / "tools/review_magicbrush.py",
        ROOT / "tools/edit_model_v1/data.py",
    ]
    return {
        "schema_version": 1,
        "config": config,
        "dataset_manifest_sha256": sha256(output / "manifest.json"),
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in files},
        "scope": "fixed20_train_pilot_only_no_full_or_training",
        "parameters_frozen": True,
        "seed": config["seed"],
        "models_receive_instructions": "BLIP: no; DINO: noun queries only; CLIP: class/color templates only",
        "target_caption_used": False,
    }


def key(identifier):
    return hashlib.sha256(identifier.encode()).hexdigest()


def load_image(meta):
    path = ROOT / meta["path"]
    if sha256(path) != meta["sha256"]:
        raise ValueError("Raw image hash changed")
    with Image.open(path) as image:
        return image.convert("RGB")


def read_cache(path, fingerprint, identifier):
    row = json.loads(path.read_text())
    if row["identity_sha256"] != fingerprint or row["id"] != identifier:
        raise ValueError(f"Cache provenance differs: {path}")
    return row


def cached_write(path, fingerprint, identifier, **values):
    fixed(path, encode({"identity_sha256": fingerprint, "id": identifier, **values}))


def serialize_evidence(evidence):
    result = {k: v for k, v in evidence.items() if k != "change_mask"}
    result["change_mask_shape"] = list(evidence["change_mask"].shape)
    result["change_mask_packbits_base64"] = base64.b64encode(np.packbits(evidence["change_mask"]).tobytes()).decode(
        "ascii"
    )
    return result


def deserialize_evidence(evidence):
    shape = evidence["change_mask_shape"]
    bits = np.frombuffer(base64.b64decode(evidence["change_mask_packbits_base64"], validate=True), dtype=np.uint8)
    mask = np.unpackbits(bits)[: int(np.prod(shape))].reshape(shape).astype(bool)
    if int(mask.sum()) != evidence["changed_pixels"]:
        raise ValueError("Changed-pixel mask is inconsistent")
    return {**evidence, "change_mask": mask}


def infer(config, output, wandb_mode):
    """Never invoked by preparation, compilation or tests."""
    import torch
    import transformers

    from tools.magicbrush_grounded_v1.models import caption_images, detect_images, verify_pairs

    if not torch.cuda.is_available():
        raise ValueError("User-run GPU inference requires CUDA; --stage plan is CPU-only")
    if sha256(ROOT / config["clip_weights"]) != config["clip_sha256"]:
        raise ValueError("Frozen CLIP weights differ")
    run_identity = identity(config, output)
    run_identity["runtime"] = {
        "torch": str(torch.__version__),
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
    }
    fixed(output / "inference-identity.json", encode(run_identity))
    fingerprint = digest(run_identity)
    manifest = json.loads((output / "manifest.json").read_text())
    index = {r["id"]: r for r in manifest["records"]}
    rows = [index[k] for k in manifest["review_ids"]]
    if len(rows) != 20 or any(r["split"] != "train" or r["official_split"] != "train" for r in rows):
        raise ValueError("Pilot scope must remain exactly 20 official/project train pairs")
    tracker = None
    if wandb_mode != "disabled":
        import wandb

        tracker = wandb.init(
            project="semantic-graph-editing",
            entity="keiranoluv-hcmus",
            mode=wandb_mode,
            name=f"magicbrush-grounded-anchor-v1-pilot-bs{config['batch_size']}-clipbs{config['clip_batch_size']}-s{config['seed']}",
            tags=[
                "magicbrush",
                "grounded-anchor-v1",
                "frozen-inference",
                "pilot-20",
                f"batch-{config['batch_size']}",
                "no-training",
            ],
            config=run_identity,
        )
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        sources = {r["source_image"]["sha256"]: r["source_image"] for r in rows}
        missing = [s for s in sorted(sources) if not (output / "captions" / f"{s}.json").exists()]
        if missing:
            captions = caption_images([load_image(sources[s]) for s in missing], config)
            for image_sha, caption in zip(missing, captions, strict=True):
                cached_write(output / "captions" / f"{image_sha}.json", fingerprint, image_sha, caption=caption)
        detector_jobs = []
        for row in rows:
            caption = read_cache(
                output / "captions" / f"{row['source_image']['sha256']}.json",
                fingerprint,
                row["source_image"]["sha256"],
            )["caption"]
            focus = mentions(scope(row["instruction"]))
            queries = focus + [q for q in caption_nouns(caption) if q not in focus]
            queries = queries[: config["max_queries"]]
            for role in ("source", "target"):
                path = output / "detections" / f"{key(row['id'])}-{role}.json"
                if path.exists():
                    previous = read_cache(path, fingerprint, row["id"])
                    if previous["queries"] != queries:
                        raise ValueError("Cached detector query selection changed")
                else:
                    detector_jobs.append((row, role, queries, focus, path))
        if detector_jobs:
            proposals = detect_images(
                [load_image(r[f"{role}_image"]) for r, role, _, _, _ in detector_jobs],
                [q for _, _, q, _, _ in detector_jobs],
                [f for _, _, _, f, _ in detector_jobs],
                config,
            )
            for (row, role, queries, _, path), proposal in zip(detector_jobs, proposals, strict=True):
                cached_write(path, fingerprint, row["id"], role=role, queries=queries, proposal=proposal)
        pending = []
        for row in rows:
            path = output / "evidence" / f"{key(row['id'])}.json"
            if path.exists():
                read_cache(path, fingerprint, row["id"])
            else:
                pending.append(row)
        if pending:
            proposals = [
                read_cache(output / "detections" / f"{key(r['id'])}-{role}.json", fingerprint, r["id"])["proposal"]
                for r in pending
                for role in ("source", "target")
            ]
            pairs = [(load_image(r["source_image"]), load_image(r["target_image"])) for r in pending]
            evidence = verify_pairs(pairs, proposals, config)
            for i, (row, proof) in enumerate(zip(pending, evidence, strict=True)):
                cached_write(
                    output / "evidence" / f"{key(row['id'])}.json",
                    fingerprint,
                    row["id"],
                    source=proposals[2 * i],
                    target=proposals[2 * i + 1],
                    evidence=serialize_evidence(proof),
                )
        report = compile_labels(config, output)
        if tracker:
            tracker.log(
                {
                    "pilot/pairs": 20,
                    "pilot/candidates": report["candidate_pairs"],
                    "inference/max_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
                }
            )
    finally:
        if tracker:
            tracker.finish()


def compile_labels(config, output):
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    inference_path = output / "inference-identity.json"
    run_identity = json.loads(inference_path.read_text()) if inference_path.exists() else None
    expected = identity(config, output)
    if run_identity and {k: v for k, v in run_identity.items() if k != "runtime"} != expected:
        raise ValueError("Inference source/config/manifest changed; use a fresh version")
    fingerprint = digest(run_identity) if run_identity else None
    records, problems = [], Counter()
    for row in manifest["records"]:
        entry = {k: row[k] for k in ("id", "split", "source_group_id", "instruction")}
        entry.update(eligible_for_training=False, human_review_status="pending", automatic_approval=False)
        path = output / "evidence" / f"{key(row['id'])}.json"
        if row["id"] not in manifest["review_ids"]:
            entry.update(status="not_in_pilot", problems=[])
        elif not path.exists() or run_identity is None:
            entry.update(status="pending_inference", problems=["missing_frozen_model_image_evidence"])
        else:
            cached = read_cache(path, fingerprint, row["id"])
            for role in ("source", "target"):
                load_image(row[f"{role}_image"])
            entry.update(
                analyze(
                    cached["source"],
                    cached["target"],
                    deserialize_evidence(cached["evidence"]),
                    row["instruction"],
                    config,
                )
            )
            entry["evidence_sha256"] = sha256(path)
        problems.update(entry["problems"])
        records.append(entry)
    result = {
        "schema_version": 1,
        "dataset_manifest_sha256": sha256(manifest_path),
        "records": records,
        "inference_identity_sha256": sha256(inference_path) if inference_path.exists() else None,
        "candidate_pairs": sum(r.get("passed_automatic_filters", False) for r in records),
        "problem_counts": dict(problems),
        "human_reviews_completed": 0,
        "ready_for_main_training": False,
        "minimum_valid_pairs": 300,
        "interpretation": "Strict-filter candidates are unverified pseudo labels; training stays disabled. No full inference is launched.",
    }
    # Every compilation is immutable, including partial failures/resumes.
    name = f"edits-{digest(result)[:16]}.json"
    fixed(output / name, encode(result))
    atomic_write(output / "current-compilation.json", {"file": name, "sha256": sha256(output / name)})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", required=True, choices=("prepare", "plan", "infer", "compile", "review", "serve", "audit")
    )
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--base-manifest", type=Path, default=DATA / "manifest.json")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="disabled")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--review-csv", type=Path)
    args = parser.parse_args(argv)
    config, output = settings(args.config)
    if args.stage == "prepare":
        result = prepare(config, output, args.base_manifest)
        print(json.dumps({k: v for k, v in result.items() if k != "scope_records"}, indent=2))
    elif args.stage == "plan":
        prepared = json.loads((output / "preparation.json").read_text())
        print(
            json.dumps(
                {
                    "GPU_started": False,
                    "scope": "20 train pairs only",
                    "models_in_sequence": [config["caption_model"], config["detector_model"], "CLIP ViT-B/32"],
                    "batch_size": config["batch_size"],
                    "clip_batch_size": config["clip_batch_size"],
                    "seed": config["seed"],
                    "grammar_supported_train": prepared["grammar_supported_train"],
                    "output": str(output),
                    "automatic_full_inference": False,
                    "automatic_training": False,
                },
                indent=2,
            )
        )
    elif args.stage == "infer":
        with exclusive(output):
            infer(config, output, args.wandb_mode)
    elif args.stage == "compile":
        result = compile_labels(config, output)
        print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))
    else:
        from tools.magicbrush_grounded_v1.review import audit, build, serve

        if args.stage == "review":
            build(config, output)
        elif args.stage == "serve":
            serve(config, args.port)
        else:
            if args.review_csv is None:
                parser.error("--stage audit requires the actual --review-csv saved by the human reviewer")
            print(json.dumps(audit(config, output, args.review_csv), indent=2))


if __name__ == "__main__":
    main()
