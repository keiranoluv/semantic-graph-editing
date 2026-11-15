"""CLI with CPU preflight, real CPU smoke, inference, review, compatibility audit."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import random
import shutil
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from src.editing.operations import EditGraph

from . import VERSION
from .backend import RELTR_COMMIT, add_attributes, infer_reltr, load_clip, load_reltr, model_identity, verify_repo
from .compatibility import audit_records, load_task_b
from .core import (
    DEFAULT_CONFIG,
    DEFAULT_LOCKED,
    ROOT,
    digest,
    graph_from_raw,
    load_pilot,
    make_record,
    outside_locked,
    read,
    sha256,
    validate_record,
    write,
)
from .review import apply_review, render_review

CACHE = ROOT / "data/cache/level3-image-graph-v1"
RELTR_URL = (
    "https://drive.usercontent.google.com/download?id=1id6oD_iwiNDD6HyCn2ORgRTIKkPD3tUD&export=download&confirm=t"
)


def get_config(path):
    cfg = read(path)
    if (
        cfg["schema"] != VERSION
        or cfg["batch_size"] != 1
        or cfg["training"] is not False
        or cfg["automatic_queue"] is not False
    ):
        raise ValueError("invalid_inference_only_config")
    for key in ("object_threshold", "relation_threshold", "merge_iou", "attribute_min_cosine", "attribute_min_margin"):
        if not isinstance(cfg[key], (int, float)) or not 0 < cfg[key] < 1:
            raise ValueError("invalid_threshold:" + key)
    if cfg["threads"] < 1 or cfg["reltr_short_side"] < 32 or cfg["orientation"] != "stored_pixels_no_exif_transpose":
        raise ValueError("invalid_preprocessing_config")
    for labels in cfg["attribute_vocabulary"].values():
        if (
            len(labels) < 2
            or len(set(labels)) != len(labels)
            or any(not isinstance(s, str) or not s.strip() for s in labels)
        ):
            raise ValueError("invalid_attribute_vocabulary")
    return cfg


def prepare(args):
    """Download pinned public model assets only; never reads or uploads pilot images."""
    outside_locked(args.reltr_repo, args.locked)
    outside_locked(args.checkpoint, args.locked)
    if not args.reltr_repo.exists():
        args.reltr_repo.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "https://github.com/yrcong/RelTR.git", str(args.reltr_repo)], check=True)
        subprocess.run(["git", "-C", str(args.reltr_repo), "checkout", "--detach", RELTR_COMMIT], check=True)
    verify_repo(args.reltr_repo)
    cfg = get_config(args.config)
    if not args.checkpoint.exists():
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        partial = args.checkpoint.with_suffix(".partial")
        with partial.open("xb") as out, urllib.request.urlopen(RELTR_URL, timeout=120) as response:
            shutil.copyfileobj(response, out)
        if sha256(partial) != cfg["reltr_checkpoint_sha256"]:
            raise ValueError("downloaded_checkpoint_hash_mismatch")
        partial.rename(args.checkpoint)
    if sha256(args.checkpoint) != cfg["reltr_checkpoint_sha256"]:
        raise ValueError("reltr_checkpoint_hash_mismatch")
    print(json.dumps({"prepared": str(args.checkpoint), "reltr_commit": RELTR_COMMIT, "gpu_used": False}))


def preflight(args):
    cfg = get_config(args.config)
    manifest, cohort = load_pilot(args.locked)
    if (
        cohort["manifest_sha256"] != cfg["locked_manifest_sha256"]
        or cohort["tasks_sha256"] != cfg["locked_tasks_sha256"]
    ):
        raise ValueError("unexpected_locked_pilot_revision")
    identity, vocab = model_identity(
        args.reltr_repo, args.checkpoint, args.clip_weights, cfg["reltr_checkpoint_sha256"]
    )
    checkpoint = load_task_b(args.task_b_checkpoint) if args.task_b_checkpoint else None
    return cfg, manifest, cohort, identity, vocab, checkpoint


def source_provenance():
    paths = sorted(Path(__file__).parent.glob("*.py")) + [
        Path(__file__).with_name("viewer.html"),
        ROOT / "src/editing/operations.py",
        ROOT / "tools/edit_model_v1/codec.py",
    ]
    return {
        "git_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "files_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in paths},
    }


def export_bundle(output, records, metadata, locked, checkpoint):
    for r in records:
        validate_record(r)
    write(output / "candidates.json", {"schema": VERSION, "run": metadata, "records": records})
    with (output / "source_graph_tasks.jsonl").open("x", encoding="utf-8") as out:
        for record in records:
            out.write(
                json.dumps(
                    {
                        "image_id": record["image_id"],
                        "candidate_id": record["candidate_id"],
                        "image_path": record["image"]["image_path"],
                        "source_sha256": record["source_sha256"],
                        "source_graph": record["source_graph"],
                        "annotation_status": record["annotation_status"],
                        "candidate_record_sha256": digest(record),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    write(output / "task-b-compatibility.json", audit_records(records, checkpoint))
    render_review(output, records, locked)
    write(
        output / "receipt.json",
        {
            "schema": VERSION,
            "complete": True,
            "pilot_inference_complete": metadata["mode"] == "infer" and len(records) == 30,
            "image_count": len(records),
            "human_accepted_count": 0,
            "files_sha256": {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()},
        },
    )


def infer(args):
    output = outside_locked(args.output, args.locked)
    if output.exists():
        raise FileExistsError("immutable_output_exists; choose a new --output version")
    cfg, manifest, cohort, identity, vocab, task_b = preflight(args)
    import torch

    device = "cpu" if args.command == "smoke" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no automatic CPU fallback")
    torch.set_num_threads(cfg["threads"])
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    if device == "cuda":
        torch.cuda.manual_seed_all(cfg["seed"])
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    packages = {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "Pillow", "scipy", "clip")}
    metadata = {
        "mode": args.command,
        "config": cfg,
        "config_sha256": sha256(args.config),
        "cohort": cohort,
        "models": identity,
        "source": source_provenance(),
        "packages": packages,
        "device": device,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_sequence": ["RelTR VG ResNet50", "CLIP ViT-B/32 region attributes"],
        "batch_size": 1,
        "seed": cfg["seed"],
        "automatic_queue": False,
        "training": False,
        "synthetic_predictions": False,
        "task_b_checkpoint_sha256": task_b["checkpoint_sha256"] if task_b else None,
    }
    output.mkdir(parents=True)
    write(output / "run-start.json", metadata)
    write(output / "vocabulary.json", vocab)
    rows = manifest["images"][:1] if args.command == "smoke" else manifest["images"]
    model = load_reltr(args.reltr_repo, args.checkpoint, cfg["reltr_checkpoint_sha256"], device)
    records = []
    for i, row in enumerate(rows):
        with Image.open(args.locked / row["image_path"]) as image:
            image = image.convert("RGB")
            raw = infer_reltr(model, image, vocab, device, cfg["reltr_short_side"])
        raw_path = output / "raw" / (row["image_id"] + ".json")
        write(raw_path, {"candidate_id": row["image_id"], "source_sha256": row["sha256"], **raw})
        graph, grounding = graph_from_raw(raw, row, cfg)
        record = make_record(
            row,
            graph,
            grounding,
            {
                "run_sha256": digest(metadata),
                "raw_prediction_sha256": sha256(raw_path),
                "raw_prediction_path": str(raw_path.relative_to(output)),
                "coordinate_system": "xyxy_pixels_stored_image",
                "attribute_status": "pending_clip",
            },
        )
        records.append(record)
        print(
            json.dumps(
                {
                    "stage": "reltr",
                    "completed": i + 1,
                    "total": len(rows),
                    "nodes": len(graph.nodes),
                    "edges": len(graph.edges),
                }
            ),
            flush=True,
        )
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    model, preprocess = load_clip(args.clip_weights, device)
    for i, record in enumerate(records):
        graph = EditGraph.from_dict(record["source_graph"])
        with Image.open(args.locked / record["image"]["image_path"]) as image:
            graph, evidence = add_attributes(
                model, preprocess, image.convert("RGB"), graph, record["grounding"], cfg, device
            )
        attr_path = output / "attributes" / (record["candidate_id"] + ".json")
        write(attr_path, evidence)
        record["source_graph"] = graph.to_dict()
        record["provenance"].update(
            attribute_status="clip_candidates_require_review", attribute_evidence_sha256=sha256(attr_path)
        )
        validate_record(record)
        print(json.dumps({"stage": "clip_attributes", "completed": i + 1, "total": len(rows)}), flush=True)
    del model
    gc.collect()
    # Recheck all 30 source byte identities after inference, before final receipt.
    if load_pilot(args.locked)[1] != cohort:
        raise ValueError("locked_inputs_changed_during_inference")
    export_bundle(output, records, metadata, args.locked, task_b)
    print(json.dumps({"output": str(output), "images": len(records), "mode": args.command, "human_reviewed": False}))


def read_bundle(directory):
    directory = Path(directory)
    receipt = read(directory / "receipt.json")
    if receipt.get("complete") is not True:
        raise ValueError("incomplete_inference_bundle")
    for name, checksum in receipt["files_sha256"].items():
        path = (directory / name).resolve()
        if directory.resolve() not in path.parents or sha256(path) != checksum:
            raise ValueError("candidate_bundle_modified")
    payload = read(directory / "candidates.json")
    if payload["schema"] != VERSION:
        raise ValueError("unexpected_candidate_schema")
    return payload


def review(args):
    output = outside_locked(args.output, args.locked)
    outside_locked(output, args.bundle)
    if output.exists():
        raise FileExistsError("review_output_must_be_new")
    payload = read_bundle(args.bundle)
    records, decisions = apply_review(payload["records"], read(args.submission))
    checkpoint = load_task_b(args.task_b_checkpoint) if args.task_b_checkpoint else None
    # Verify the reviewed cohort against current immutable images before writes.
    _, cohort = load_pilot(args.locked)
    if cohort != payload["run"]["cohort"]:
        raise ValueError("review_cohort_does_not_match_inference")
    # Validate before writing any review outputs.
    output.mkdir(parents=True)
    write(
        output / "reviewed.json",
        {
            "schema": VERSION,
            "records": records,
            "decisions": decisions,
            "candidate_bundle_sha256": sha256(args.bundle / "candidates.json"),
            "submission_sha256": sha256(args.submission),
        },
    )
    write(output / "task-b-compatibility.json", audit_records(records, checkpoint))
    with (output / "source_graph_tasks.jsonl").open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    if records:
        render_review(output, records, args.locked)
    write(
        output / "receipt.json",
        {
            "schema": VERSION,
            "complete": True,
            "human_accepted_count": len(records),
            "files_sha256": {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()},
        },
    )
    print(json.dumps({"accepted": len(records), "total": len(decisions), "output": str(output)}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "preflight", "smoke", "infer", "review", "compatibility"):
        p = sub.add_parser(command)
        p.add_argument("--locked", type=Path, default=DEFAULT_LOCKED)
        p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        p.add_argument("--reltr-repo", type=Path, default=CACHE / "RelTR")
        p.add_argument("--checkpoint", type=Path, default=CACHE / "checkpoint0149.pth")
        p.add_argument(
            "--clip-weights", type=Path, default=ROOT / "data/cache/vg150-clip-vit-b32-v1/weights/ViT-B-32.pt"
        )
        p.add_argument(
            "--task-b-checkpoint", type=Path, default=ROOT / "runs/level2/emnlp2020-task-b-wandb-v4/gnn-seed42/best.pt"
        )
        if command in {"infer", "smoke", "review", "compatibility"}:
            p.add_argument("--output", type=Path, required=True)
        if command == "infer":
            p.add_argument("--device", choices=("cpu", "cuda"), required=True)
        if command in {"review", "compatibility"}:
            p.add_argument("--bundle", type=Path, required=True)
        if command == "review":
            p.add_argument("--submission", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(args)
    elif args.command == "preflight":
        _, _, cohort, identity, vocab, checkpoint = preflight(args)
        print(
            json.dumps(
                {
                    "passed": True,
                    "cohort": cohort,
                    "models": identity,
                    "vocabulary_sizes": {k: len(v) for k, v in vocab.items()},
                    "task_b_checkpoint_sha256": checkpoint["checkpoint_sha256"] if checkpoint else None,
                    "gpu_used": False,
                },
                ensure_ascii=False,
            )
        )
    elif args.command in {"infer", "smoke"}:
        infer(args)
    elif args.command == "review":
        review(args)
    else:
        payload = read_bundle(args.bundle)
        outside_locked(args.output, args.locked)
        write(args.output, audit_records(payload["records"], load_task_b(args.task_b_checkpoint)))


if __name__ == "__main__":
    main()
