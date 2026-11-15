"""Immutable five-image runs, bounded one-image smoke, manual evaluation."""

import argparse
import fcntl
import importlib.metadata
import json
import os
import random
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tools.level3_image_graph_v1.core import DEFAULT_LOCKED, ROOT, digest, outside_locked, read, sha256, write
from tools.level3_sgg_compare_v1.runner import BASELINE
from tools.level3_sgg_compare_v1.runner import CONFIG as OLD_CONFIG
from tools.level3_sgg_compare_v1.runner import inputs

from . import VERSION
from .adapter import adapt
from .assets import CACHE, LOCK, prepare, verify
from .comparison import export
from .evaluation import evaluate

CONFIG = ROOT / "configs/level3/sgg-pretrained-v2.json"


def verified_bundle(path, filename):
    receipt = read(path / "receipt.json")
    if not receipt["execution_complete"]:
        raise ValueError("incomplete_run")
    for name, checksum in receipt["files_sha256"].items():
        target = (path / name).resolve()
        if path.resolve() not in target.parents or sha256(target) != checksum:
            raise ValueError("run_artifact_changed:" + name)
    if filename not in receipt["files_sha256"]:
        raise ValueError("records_not_bound_to_receipt")
    return read(path / filename)


def cohort(cfg):
    if (
        cfg["schema"] != VERSION
        or cfg["training"] is not False
        or cfg["automatic_queue"] is not False
        or cfg["batch_size"] != 1
        or cfg["seed"] != 42
    ):
        raise ValueError("invalid_protocol")
    if sha256(OLD_CONFIG) != cfg["cohort_config_sha256"]:
        raise ValueError("parent_cohort_changed")
    parent, rows, _, identity = inputs(SimpleNamespace(config=OLD_CONFIG, locked=DEFAULT_LOCKED, baseline=BASELINE))
    if cfg["images"] != parent["images"]:
        raise ValueError("cohort_mismatch")
    robin_path = ROOT / cfg["robin_run"]
    if sha256(robin_path / "receipt.json") != cfg["robin_receipt_sha256"]:
        raise ValueError("robin_baseline_changed")
    robin = verified_bundle(robin_path, "robin.json")
    if isinstance(robin, dict):
        robin = robin["records"]
    if len(robin) != 5 or {r["candidate_id"] for r in robin} != {r["image_id"] for r in rows}:
        raise ValueError("robin_cohort_mismatch")
    return rows, robin, identity


def new_output(path, cfg, cache=CACHE):
    path = path.resolve()
    for protected in (
        ROOT / "data",
        ROOT / "src",
        ROOT / "notebooks",
        ROOT / "runs/level3/sgg-compare-v1",
        BASELINE,
        ROOT / cfg["robin_run"],
        cache,
    ):
        outside_locked(path, protected)
    if path.exists():
        raise FileExistsError("immutable_output_exists; choose a new version")
    path.mkdir(parents=True)
    return path


def code_identity():
    files = list(Path(__file__).parent.glob("*.py")) + [
        CONFIG,
        LOCK,
        ROOT / "requirements-level3-sgg-pretrained-v2.txt",
    ]
    for folder in ("tools/level3_sgg_compare_v1", "tools/level3_image_graph_v1"):
        files += list((ROOT / folder).glob("*.py"))
    files += [ROOT / "src/editing/operations.py"]
    return {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "files_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sorted(files)},
    }


def receipt(output, mode, success):
    write(
        output / "receipt.json",
        {
            "schema": VERSION,
            "mode": mode,
            "execution_complete": True,
            "all_records_ok": success,
            "files_sha256": {
                str(p.relative_to(output)): sha256(p)
                for p in sorted(output.rglob("*"))
                if p.is_file() and "wandb" not in p.relative_to(output).parts
            },
        },
    )


def run(args):
    cfg = read(CONFIG)
    rows, robin, identity = cohort(cfg)
    assets = verify(args.cache)
    if args.command == "preflight":
        print(
            json.dumps(
                {
                    "status": "ok",
                    "images": [r["image_id"] for r in rows],
                    "cohort": identity,
                    "assets_sha256": digest(assets),
                }
            )
        )
        return 0
    import torch

    from .backends import React, Relate, detect_owl

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    selected = rows[:1] if args.command == "smoke" else rows
    output = new_output(args.output, cfg, args.cache)
    (output / "native").mkdir()
    provenance = {
        "schema": VERSION,
        "model": args.model,
        "mode": args.command,
        "config": cfg,
        "cohort": identity,
        "assets": assets,
        "code": code_identity(),
        "device": args.device,
        "runtime": {
            p: importlib.metadata.version(p)
            for p in ("torch", "torchvision", "transformers", "onnxruntime-gpu", "numpy")
        },
        "cuda_build": torch.version.cuda,
        "seed": 42,
        "batch_size": 1,
        "automatic_queue": False,
        "selected_images": [r["image_id"] for r in selected],
    }
    if args.device == "cuda":
        provenance["gpu"] = torch.cuda.get_device_name()
    write(output / "run.json", provenance)
    write(output / "environment.json", {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()})
    records = []
    tracking = None
    model_name = "react++" if args.model == "react" else "relate-anything"
    try:
        if args.command == "infer":
            import wandb

            if os.environ.get("WANDB_MODE", "online") != "online":
                raise ValueError("full_pilot_requires_WANDB_MODE_online; smoke_needs_no_wandb")
            tracking = wandb.init(
                project="level3-sgg-pretrained-v2", name=output.name, config=provenance, dir=str(output), mode="online"
            )
            write(output / "wandb-run.json", {"id": tracking.id, "url": tracking.url})
        detections = None
        if args.model == "react":
            backend = React(args.cache, args.device, output, cfg["react"])
        else:
            detections = detect_owl(args.cache, args.device, selected, DEFAULT_LOCKED, output, cfg["relate"])
            write(output / "detections.json", detections)
            backend = Relate(args.cache, args.device, output, cfg["relate"])
        write(output / "model-metadata.json", backend.metadata)
        for row in selected:
            start = time.perf_counter()
            native_path = output / "native" / (row["image_id"] + ".npz")
            native = None
            try:
                if detections is None:
                    objects, relations, native = backend.predict(DEFAULT_LOCKED / row["image_path"], native_path)
                else:
                    objects = detections[row["image_id"]]["objects"]
                    relations, native = backend.predict(DEFAULT_LOCKED / row["image_path"], objects, native_path)
                    native["detector"] = detections[row["image_id"]]["detector_native"]
                write(
                    output / "native" / (row["image_id"] + ".json"),
                    {"objects": objects, "relations": relations, "native": native},
                )
                rec = adapt(row, model_name, objects, relations, provenance, native)
            except (ValueError, KeyError, IndexError) as error:
                rec = {
                    "candidate_id": row["image_id"],
                    "image_id": row["image_id"],
                    "image": row,
                    "source_sha256": row["sha256"],
                    "model": model_name,
                    "status": "parse_error",
                    "error": repr(error),
                    "native_output": native,
                    "provenance": provenance,
                }
            rec["elapsed_seconds"] = time.perf_counter() - start
            records.append(rec)
            if tracking:
                tracking.log(
                    {
                        "image_index": len(records),
                        "parse_success": int(rec["status"] == "ok"),
                        "inference_seconds": rec["elapsed_seconds"],
                    }
                )
            print(
                json.dumps(
                    {"image": row["image_id"], "model": model_name, "status": rec["status"]}, ensure_ascii=False
                ),
                flush=True,
            )
        write(output / "execution.json", {**backend.close(), "torch_cuda_initialized": torch.cuda.is_initialized()})
        write(output / "records.json", records)
        export(rows, {"robin-sam": robin, model_name: records}, DEFAULT_LOCKED, output)
        ok = all(r["status"] == "ok" for r in records)
        if tracking:
            tracking.summary["parse_success"] = sum(r["status"] == "ok" for r in records) / len(records)
            tracking.summary["quality_status"] = "requires_human_review"
            tracking.finish(exit_code=0 if ok else 2)
            tracking = None
        receipt(output, args.command, ok)
        return 0 if ok else 2
    except Exception as error:
        if tracking:
            tracking.finish(exit_code=1)
        write(
            output / "failure.json", {"error": repr(error), "completed_records": records, "execution_complete": False}
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "preflight", "smoke", "infer"):
        p = sub.add_parser(command)
        p.add_argument("--cache", type=Path, default=CACHE)
        if command in ("smoke", "infer"):
            p.add_argument("--model", choices=("react", "relate"), required=True)
            p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
            p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("compare")
    p.add_argument("--react-run", type=Path)
    p.add_argument("--relate-run", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("evaluate")
    p.add_argument("--comparison", type=Path, required=True)
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.cache)
        return 0
    if args.command == "evaluate":
        # Metrics file only; refuse any path within preserved data/baselines.
        for path in (ROOT / "data", ROOT / "src", ROOT / "notebooks", ROOT / read(CONFIG)["robin_run"], BASELINE):
            outside_locked(args.output, path)
        write(args.output, evaluate(read(args.comparison), read(args.review)))
        return 0
    if args.command == "compare":
        cfg = read(CONFIG)
        rows, robin, _ = cohort(cfg)
        models = {"robin-sam": robin}
        for model, path in [("react++", args.react_run), ("relate-anything", args.relate_run)]:
            if path:
                models[model] = verified_bundle(path, "records.json")
                if any(r["model"] != model for r in models[model]):
                    raise ValueError("run_model_mismatch")
        output = new_output(args.output, cfg)
        export(rows, models, DEFAULT_LOCKED, output)
        receipt(output, "comparison", True)
        return 0
    if args.command == "preflight":
        return run(args)
    # Same lock as ROBIN launchers; CLI also locks, no concurrent vision jobs here.
    lock = ROOT / "data/cache/level3-sgg-v1/inference.lock"
    with lock.open("a") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run(args)
