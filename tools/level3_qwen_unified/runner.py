"""Single entry point: collect completed cases, resume pending inference, or serve the 200-case demo."""

import argparse
import functools
import http.server
import json
import os
import signal
from pathlib import Path

from PIL import Image, ImageOps

from tools.level3_image_graph_v1.core import ROOT, sha256, write
from tools.level3_qwen21_v1.core import finish

from .core import GPU_LOCK, atomic, collect, identity, inputs, load_case, lock, save_state, workspace
from .viewer import render


def attempt_dir(parent):
    parent.mkdir(parents=True, exist_ok=True)
    i = 1
    while (parent / f"attempt-{i:04d}").exists():
        i += 1
    path = parent / f"attempt-{i:04d}"
    path.mkdir(exist_ok=False)
    return path


def infer_pending(out, cfg, tasks, state, code, editor_factory=None, tracker_factory=None):
    """One bounded pass. Successes survive interruptions; retries get new attempt directories."""
    if state["successful"] == len(tasks):
        return 0  # no model, CUDA, weights or W&B needed when already complete
    if os.environ.get("WANDB_MODE", "online") != "online":
        raise ValueError("online_WandB_required")
    if editor_factory is None:
        from .backend import Editor

        editor_factory = Editor
    if tracker_factory is None:
        import wandb

        tracker_factory = wandb.init
    attempt = attempt_dir(out / "attempts")
    write(
        attempt / "run.json",
        {
            "code": code,
            "config": cfg,
            "reused_successes": state["successful"],
            "pending_task_ids": [e["task_id"] for e in state["entries"] if e["status"] != "ok"],
        },
    )
    known_sources = {e["image_id"]: e["result"]["source_path"] for e in state["entries"] if e["status"] == "ok"}
    editor = tracking = case = None
    error = None
    generated = 0
    try:
        tracking = tracker_factory(
            project=cfg["wandb_project"],
            name=out.name + "-" + attempt.name,
            config={**cfg, "code": code},
            dir=str(attempt),
            mode="online",
        )
        write(attempt / "wandb.json", {"id": tracking.id, "url": tracking.url})
        editor = editor_factory(cfg, attempt)
        for task, entry in zip(tasks, state["entries"], strict=True):
            if entry["status"] == "ok":
                continue
            if sha256(task["source_path"]) != task["sha256"]:
                raise ValueError("original_source_changed_before_inference")
            case = attempt_dir(out / "cases" / task["task_id"])
            write(case / "intent.json", task)
            source, target = case / "source.png", case / "target.png"
            if task["image_id"] in known_sources:
                os.link(known_sources[task["image_id"]], source)
            else:
                with Image.open(task["source_path"]) as im:
                    ImageOps.exif_transpose(im).convert("RGB").save(source)
            metrics = editor.generate(source, task["instruction_en"], target)
            if metrics["completed_steps"] != 40 or sha256(target) != metrics["target"]["sha256"]:
                raise ValueError("incomplete_or_invalid_generated_image")
            row = {
                "task_id": task["task_id"],
                "image_id": task["image_id"],
                "dataset": task["dataset"],
                "prompt": task["instruction_en"],
                "operation": task["operation"],
                "visual_evidence": task["visual_evidence"],
                "source_path": str(source),
                "source_sha256": sha256(source),
                "target_path": str(target),
                "native_source_sha256": task["sha256"],
                "case_dir": str(case),
                "status": "ok",
                "seed": 42,
                "dtype": "bfloat16",
                "steps_requested": 40,
                "human_quality_label": None,
                "execution_attempt": str(attempt),
                "code": code,
                **metrics,
            }
            write(case / "result.json", row)
            finish(case)
            entry.update(status="ok", **load_case(case, task))
            state["successful"] += 1
            state["execution_complete"] = state["successful"] == len(tasks)
            known_sources[task["image_id"]] = str(source)
            generated += 1
            save_state(out, state)
            render(out, cfg, tasks, state)
            tracking.log(
                {
                    "successful": state["successful"],
                    "generated_this_attempt": generated,
                    "inference_seconds": metrics["inference_seconds"],
                    "peak_reserved_mib": metrics["torch_peak_reserved_mib"],
                }
            )
            print(
                json.dumps({"successful": state["successful"], "requested": len(tasks), "task_id": task["task_id"]}),
                flush=True,
            )
            case = None
    except (Exception, KeyboardInterrupt) as exc:
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "oom": type(exc).__name__ == "OutOfMemoryError",
            "case_dir": str(case) if case else None,
        }
        write(attempt / "failure.json", error)
    finally:
        if editor is not None:
            editor.close()
        save_state(out, state)
        render(out, cfg, tasks, state)
        write(
            attempt / "execution.json",
            {
                "requested": len(tasks),
                "successful": state["successful"],
                "generated_this_attempt": generated,
                "error": error,
                "quality_labels_created": False,
            },
        )
        if tracking is not None:
            tracking.finish(exit_code=0 if error is None else 1)
        atomic(out / "last-attempt.json", {"path": str(attempt), "error": error, "generated": generated})
    return 2 if error else 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["collect", "infer", "serve", "preflight"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--import-run", action="append", type=Path, default=[])
    p.add_argument(
        "--no-import",
        action="store_true",
        help="Fresh 200-request inference; existing successes in output still resume",
    )
    p.add_argument("--port", type=int, default=8008)
    args = p.parse_args()
    cfg, tasks = inputs()
    code = identity()
    if args.command == "preflight":
        print(json.dumps({"status": "ok", "requests": len(tasks), "device": "cpu", "code_commit": code["commit"]}))
        return 0
    out = workspace(args.output, cfg, code)
    imports = ([] if args.no_import else [ROOT / r for r in cfg["default_import_runs"]]) + args.import_run
    with lock(out / ".state.lock"):
        state = collect(out, cfg, tasks, imports)
        render(out, cfg, tasks, state)
        if args.command == "infer" and not state["execution_complete"]:

            def terminate(_signum, _frame):
                raise KeyboardInterrupt("SIGTERM: keep completed case receipts")

            signal.signal(signal.SIGTERM, terminate)
            with lock(GPU_LOCK):
                # Never queue behind the running legacy inference job.
                return infer_pending(out, cfg, tasks, state, code)
    print(
        json.dumps(
            {
                "output": str(out),
                "successful": state["successful"],
                "requested": len(tasks),
                "pending": len(tasks) - state["successful"],
                "new_GPU_inference": False,
            }
        ),
        flush=True,
    )
    if args.command == "serve":
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(out))
        with http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
            print(f"http://localhost:{args.port}/comparison.html", flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
    return 0
