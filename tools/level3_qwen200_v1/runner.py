"""Generate only requests 101–200 from original sources; independent of the first 100 targets."""
import argparse
import fcntl
import importlib.metadata
import inspect
import json
import os
import signal
import time
from pathlib import Path

from PIL import Image

from tools.level3_image_graph_v1.core import ROOT, read, sha256, write
from tools.level3_qwen21_v1.core import CACHE, LOCK, finish
from tools.level3_qwen21_v1.runner import MemoryMonitor, call_args, check_image, runtime
from tools.level3_qwen100_v1.core import source_png
from tools.level3_qwen100_v1.receipt import finish_run

from .core import CONFIG, MANIFEST, identity, new_output, preflight, resume_cases
from .viewer import render


def run(args):
    code = identity()  # commit gate before touching CUDA
    cfg, tasks = preflight()
    Pipeline = runtime()
    import torch

    import wandb

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16_CUDA_required")
    if os.environ.get("WANDB_MODE", "online") != "online":
        raise ValueError("online_WandB_required")
    ident = {
        "manifest_sha256": sha256(MANIFEST),
        "config_sha256": sha256(CONFIG),
        "asset_lock_sha256": sha256(LOCK),
        "code": code,
    }
    previous = resume_cases(args.resume_from, ident, tasks) if args.resume_from else {}
    if set(previous) - {r["task_id"] for r in tasks}:
        raise ValueError("unknown_resume_case")
    out = new_output(args.output)
    write(
        out / "run.json",
        {
            **ident,
            "schema": "qwen200-extension-run-v1",
            "config": cfg,
            "device": torch.cuda.get_device_name(),
            "resume_from": str(args.resume_from) if args.resume_from else None,
            "reused_cases": list(previous),
            "main_run_user_authorized": True,
            "no_quality_retries": True,
        },
    )
    write(out / "environment.json", {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()})
    write(out / "instructions.json", read(MANIFEST))
    results = []
    error = None
    tracking = None
    monitor = MemoryMonitor()
    monitor.start()
    started = time.perf_counter()
    try:
        tracking = wandb.init(
            project=cfg["wandb_project"], name=out.name, config={**cfg, **ident}, dir=str(out), mode="online"
        )
        write(out / "wandb.json", {"id": tracking.id, "url": tracking.url})
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        t0 = time.perf_counter()
        pipe = Pipeline.from_pretrained(
            CACHE / "model", torch_dtype=torch.bfloat16, local_files_only=True, use_safetensors=True
        )
        for c in [pipe.text_encoder, pipe.transformer, pipe.vae]:
            c.eval().requires_grad_(False)
        pipe.enable_model_cpu_offload(gpu_id=0)
        pipe.vae.enable_tiling()
        pipe.set_progress_bar_config(disable=True)
        write(
            out / "pipeline.json",
            {
                "class": type(pipe).__name__,
                "scheduler": dict(pipe.scheduler.config),
                "pipeline_source_sha256": sha256(inspect.getfile(Pipeline)),
                "load_seconds": time.perf_counter() - t0,
                "memory_mode": "model_cpu_offload",
                "vae_tiling": True,
            },
        )
        for i, task in enumerate(tasks):
            tid = task["task_id"]
            if tid in previous:
                results.append(previous[tid])
                print(json.dumps({"task": tid, "reused": True}), flush=True)
                continue
            case = out / "cases" / tid
            case.mkdir(parents=True, exist_ok=False)
            write(case / "intent.json", task)
            source = case / "source.png"
            source_hash = source_png(task, source)
            row = {
                "task_id": tid,
                "image_id": task["image_id"],
                "dataset": task["dataset"],
                "prompt": task["instruction_en"],
                "operation": task["operation"],
                "request_index": task["request_index"],
                "parent_task_id": task["parent_task_id"],
                "evaluation_group": task["evaluation_group"],
                "visual_evidence": task["visual_evidence"],
                "native_source_sha256": task["sha256"],
                "source_path": str(source),
                "source_sha256": source_hash,
                "case_dir": str(case),
                "target_path": None,
                "status": "running",
                "human_quality_label": None,
                "seed": 42,
                "dtype": "bfloat16",
                "steps_requested": 40,
            }
            results.append(row)
            t0 = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            steps = []

            def progress(_p, j, _t, kwargs, steps=steps):
                steps.append(j)
                return kwargs

            with Image.open(source) as im, torch.inference_mode():
                generated = pipe(
                    **call_args(
                        {**cfg["generation"], "prompt": row["prompt"]},
                        im.convert("RGB"),
                        torch.Generator("cuda").manual_seed(42),
                    ),
                    callback_on_step_end=progress,
                )
            torch.cuda.synchronize()
            target = case / "target.png"
            generated.images[0].save(target)
            row.update(
                target_path=str(target),
                target=check_image(target),
                completed_steps=len(steps),
                inference_seconds=time.perf_counter() - t0,
                torch_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                torch_peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                gpu_memory=monitor.summary(t0),
            )
            if len(steps) != 40:
                raise ValueError("incomplete_steps")
            row["status"] = "ok"
            write(case / "result.json", row)
            finish(case)
            tracking.log(
                {
                    "case_index": i + 1,
                    "successful": len(results),
                    "inference_seconds": row["inference_seconds"],
                    "peak_reserved_mib": row["torch_peak_reserved_mib"],
                }
            )
            print(
                json.dumps(
                    {"completed": i + 1, "total": 100, "task": tid, "seconds": row["inference_seconds"], "status": "ok"}
                ),
                flush=True,
            )
    except (Exception, KeyboardInterrupt) as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "oom": isinstance(exc, torch.OutOfMemoryError)}
        if results and results[-1]["status"] == "running":
            results[-1].update(status="error", error=error)
        write(out / "failure.json", error)
    finally:
        monitor.close()
        seen = {r["task_id"] for r in results}
        for task in tasks:
            if task["task_id"] not in seen:
                results.append(
                    {
                        "task_id": task["task_id"],
                        "source_path": task["source_path"],
                        "prompt": task["instruction_en"],
                        "operation": task["operation"],
                        "request_index": task["request_index"],
                        "parent_task_id": task["parent_task_id"],
                        "evaluation_group": task["evaluation_group"],
                        "visual_evidence": task["visual_evidence"],
                        "status": "not_run",
                        "target_path": None,
                    }
                )
        write(out / "results.json", results)
        write(
            out / "execution.json",
            {
                "successful": sum(r["status"] == "ok" for r in results),
                "requested": 100,
                "error": error,
                "total_seconds": time.perf_counter() - started,
                "gpu_memory": monitor.summary(),
                "memory_samples": monitor.samples,
                "quality_labels_created": False,
            },
        )
        render(results, out)
        if tracking:
            tracking.finish(exit_code=0 if error is None else 1)
        # W&B files may continue changing; freeze experiment files only.
        finish_run(out, success=error is None and all(r["status"] == "ok" for r in results))
    print(
        json.dumps({"output": str(out), "successful": sum(r["status"] == "ok" for r in results), "error": error}),
        flush=True,
    )
    return 0 if error is None else 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    check = sub.add_parser("preflight")
    check.add_argument("--skip-assets", action="store_true", help="Verify manifests and original images only")
    a = sub.add_parser("authoring")
    a.add_argument("--output", type=Path, required=True)
    a = sub.add_parser("infer")
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--resume-from", type=Path)
    args = p.parse_args()
    if args.command == "preflight":
        cfg, tasks = preflight(not args.skip_assets)
        print(
            json.dumps(
                {
                    "status": "ok",
                    "new_requests": len(tasks),
                    "cohort_requests": 200,
                    "color_probes": 10,
                    "batch": 1,
                    "seed": 42,
                    "steps": 40,
                    "GPU_inference": False,
                }
            )
        )
        return 0
    if args.command == "authoring":
        cfg, tasks = preflight(False)
        out = new_output(args.output)
        render(tasks, out, authoring=True)
        write(out / "instructions.json", read(MANIFEST))
        finish(out)
        print(out / "instructions.html")
        return 0

    # Same GPU lock as the existing Qwen/REACT launchers. No automatic next run.
    def terminate(_signal, _frame):
        raise KeyboardInterrupt("SIGTERM: preserving successful case receipts")

    signal.signal(signal.SIGTERM, terminate)
    with (ROOT / "data/cache/level3-sgg-v1/inference.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run(args)
