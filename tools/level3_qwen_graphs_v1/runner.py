"""One frozen REACT++ model, 100 paired sources/targets; no quality scoring."""

import argparse
import fcntl
import importlib.metadata
import json
import os
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np

from tools.level3_image_graph_v1.core import ROOT, sha256, write
from tools.level3_qwen21_v1.core import finish
from tools.level3_qwen21_v1.runner import MemoryMonitor
from tools.level3_qwen100_v1.receipt import finish_run
from tools.level3_sgg_pretrained_v2.adapter import adapt
from tools.level3_sgg_pretrained_v2.assets import CACHE
from tools.level3_sgg_pretrained_v2.backends import React

from .core import CONFIG, code_identity, inputs, new_output, normalize, verify_react
from .viewer import render


def preflight():
    cfg, rows = inputs()
    assets = verify_react()
    import onnxruntime as ort
    import torch

    if torch.version.cuda != "12.8" or "sm_120" not in torch._C._cuda_getArchFlags():
        raise ValueError("CUDA128_Blackwell_wheel_required")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise ValueError("ORT_CUDA_provider_unavailable")
    return cfg, rows, assets


def infer(args):
    cfg, rows, assets = preflight()
    code = code_identity()
    if os.environ.get("WANDB_MODE", "online") != "online":
        raise ValueError("online_WandB_required")
    import torch

    import wandb

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = new_output(args.output)
    provenance = {"config": cfg, "config_sha256": sha256(CONFIG), "assets": assets, "code": code, "device": "cuda"}
    write(out / "run.json", provenance)
    write(out / "environment.json", {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()})
    bound = {"run_json_sha256": sha256(out / "run.json"), "run_path": str(out), "model": "react++"}
    results = [
        {
            "task_id": r["task_id"],
            "image_id": r["image_id"],
            "dataset": r["dataset"],
            "prompt": r["prompt"],
            "roles": {
                role: {"status": "not_run", "native_image_path": r[role + "_path"], "graph_path": None}
                for role in ("source", "target")
            },
        }
        for r in rows
    ]
    monitor = MemoryMonitor()
    monitor.start()
    started = time.perf_counter()
    backend, tracking, error, profile = None, None, None, None
    try:
        tracking = wandb.init(
            project=cfg["wandb_project"], name=out.name, config=provenance, dir=str(out), mode="online"
        )
        write(out / "wandb.json", {"id": tracking.id, "url": tracking.url})
        backend = React(CACHE, "cuda", out, cfg["react"])
        write(out / "model-metadata.json", backend.metadata)
        write(
            out / "runtime.json",
            {
                "providers": backend.sess.get_providers(),
                "gpu": torch.cuda.get_device_name(),
                "cuda_build": torch.version.cuda,
                "inputs": [{"name": i.name, "shape": i.shape, "type": i.type} for i in backend.sess.get_inputs()],
                "preprocess": cfg["preprocess"],
            },
        )
        count = 0
        for pair in results:
            case = out / "cases" / pair["task_id"]
            case.mkdir(parents=True, exist_ok=False)
            for role, item in pair["roles"].items():
                image = normalize(item["native_image_path"], case / (role + "-input.png"))
                image["image_id"] = pair["task_id"] + "--" + role
                item.update(input=image, status="running")
                raw = case / (role + "-native.npz")
                t0 = time.perf_counter()
                try:
                    objects, relations, native = backend.predict(Path(image["image_path"]), raw)
                    write(
                        case / (role + "-decoded.json"), {"objects": objects, "relations": relations, "native": native}
                    )
                    record = adapt(image, "react++", objects, relations, bound, native)
                except (ValueError, KeyError, IndexError) as exc:
                    record = {
                        "status": "parse_error",
                        "error": repr(exc),
                        "image": image,
                        "provenance": bound,
                        "native_output": {"raw_npz": raw.name if raw.exists() else None},
                    }
                record.update(pair_task_id=pair["task_id"], image_role=role, elapsed_seconds=time.perf_counter() - t0)
                graph_path = case / (role + "-graph.json")
                write(graph_path, record)
                item.update(
                    status=record["status"], graph_path=str(graph_path), elapsed_seconds=record["elapsed_seconds"]
                )
                count += 1
                tracking.log(
                    {
                        "image_index": count,
                        "parse_success": int(record["status"] == "ok"),
                        "seconds": record["elapsed_seconds"],
                    }
                )
                print(
                    json.dumps(
                        {"image": count, "total": 200, "task": pair["task_id"], "role": role, "status": item["status"]}
                    ),
                    flush=True,
                )
            write(case / "pair.json", pair)
            finish(case, success=all(x["status"] == "ok" for x in pair["roles"].values()))
    except (Exception, KeyboardInterrupt) as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "oom": "out of memory" in str(exc).lower()}
    finally:
        if backend is not None:
            try:
                profile = backend.close()
            except Exception as exc:
                error = error or {"type": type(exc).__name__, "message": str(exc), "stage": "CUDA_profile"}
        monitor.close()
        for pair in results:
            for item in pair["roles"].values():
                if item["status"] == "running":
                    item.update(status="error", error=error)
        statuses = Counter(item["status"] for pair in results for item in pair["roles"].values())
        ok = statuses["ok"] == 200 and error is None
        summary = {
            "requested_pairs": 100,
            "requested_images": 200,
            "status_counts": dict(statuses),
            "parse_success": statuses["ok"] / 200,
            "error": error,
            "seconds_including_load_inference_and_profile": time.perf_counter() - started,
            "gpu_memory": monitor.summary(),
            "profile": profile,
            "quality_metrics": {
                "object_precision_recall_f1": None,
                "relation_precision_recall_f1": None,
                "grounding_accuracy": None,
                "hallucination_rate": None,
                "graph_consistency": None,
            },
            "quality_status": "requires_graph_ground_truth_or_manual_review",
            "attributes": "unsupported_no_attribute_head",
            "cross_image_alignment": "not_computed",
        }
        write(out / "results.json", results)
        write(out / "execution.json", summary)
        write(out / "memory-samples.json", monitor.samples)
        if error:
            write(out / "failure.json", error)
        render(results, out)
        if tracking:
            tracking.summary["parse_success"] = summary["parse_success"]
            tracking.summary["quality_status"] = summary["quality_status"]
            tracking.finish(exit_code=0 if ok else 1)
        finish_run(out, success=ok)
    print(json.dumps({"output": str(out), **summary}), flush=True)
    return 0 if ok else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    p = sub.add_parser("infer")
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        cfg, rows, _ = preflight()
        print(
            json.dumps(
                {
                    "status": "ok",
                    "pairs": len(rows),
                    "images": 2 * len(rows),
                    "react": cfg["react"],
                    "seed": 42,
                    "batch_size": 1,
                }
            )
        )
        return 0
    with (ROOT / "data/cache/level3-sgg-v1/inference.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return infer(args)
