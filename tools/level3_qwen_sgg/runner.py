"""Collect/resume/serve one SGG cohort, with a hard ceiling on new GPU predictions."""

import argparse
import functools
import http.server
import json
import os
import signal
import time
from collections import Counter
from pathlib import Path

from tools.level3_image_graph_v1.core import sha256, write
from tools.level3_qwen21_v1.core import finish
from tools.level3_qwen_graphs_v1.core import normalize
from tools.level3_qwen_unified.core import GPU_LOCK, atomic, lock
from tools.level3_qwen_unified.runner import attempt_dir
from tools.level3_sgg_pretrained_v2.adapter import adapt

from .core import accept, collect, identity, inputs, pair_results, prediction, protocol, save_index, workspace
from .viewer import render


def refresh(out, cfg, rows, images):
    save_index(out, cfg, rows, images)
    pairs = pair_results(out, rows, images)
    render(pairs, images, out)
    atomic(
        out / "snapshot.json",
        {
            "schema": "qwen-sgg-snapshot-v1",
            "complete": all(i["status"] == "ok" for i in images.values()),
            "files_sha256": {
                n: sha256(out / n)
                for n in ("run.json", "index.json", "results.json", "review-data.json", "comparison.html")
            },
            "pair_receipts": {
                r["task_id"]: sha256(out / "pairs" / r["task_id"] / "receipt.json")
                for r in pairs
                if all(i["status"] == "ok" for i in r["roles"].values())
            },
        },
    )


def infer_pending(out, cfg, rows, images, code, max_new, backend_factory=None, tracker_factory=None):
    pending = [i for i in images.values() if i["status"] != "ok"]
    if not pending:
        return 0
    if len(pending) > max_new:
        raise ValueError(f"new_image_budget_exceeded:{len(pending)}>{max_new}")
    if os.environ.get("WANDB_MODE", "online") != "online":
        raise ValueError("online_WandB_required")
    if backend_factory is None:
        from .backend import Editor

        backend_factory = Editor
    if tracker_factory is None:
        import wandb

        tracker_factory = wandb.init
    attempt = attempt_dir(out / "attempts")
    provenance = {
        "code": code,
        "config": cfg,
        "protocol": protocol(cfg),
        "device": "cuda",
        "pending_native_sha256": [i["native_sha256"] for i in pending],
        "requested_new_images": len(pending),
        "reused_unique_images": len(images) - len(pending),
        "max_new_images": max_new,
    }
    write(attempt / "run.json", provenance)
    backend = tracking = case = None
    error, runtime = None, {}
    generated = attempted = 0
    started = time.perf_counter()
    try:
        tracking = tracker_factory(
            project=cfg["wandb_project"],
            name=out.name + "-" + attempt.name,
            config=provenance,
            dir=str(attempt),
            mode="online",
        )
        write(attempt / "wandb.json", {"id": tracking.id, "url": tracking.url})
        backend = backend_factory(cfg, attempt)
        for item in pending:
            if sha256(item["native_path"]) != item["native_sha256"]:
                raise ValueError("native_image_changed_before_inference")
            case = attempt_dir(out / "predictions" / item["native_sha256"])
            image = normalize(item["native_path"], case / "input.png")
            image["image_id"] = "sgg-" + item["native_sha256"]
            attempted += 1
            objects, relations, native, metrics = backend.predict(Path(image["image_path"]), case / "native.npz")
            write(case / "decoded.json", {"objects": objects, "relations": relations, "native": native})
            rec = adapt(
                image,
                "react++",
                objects,
                relations,
                {
                    "model": "react++",
                    "run_path": str(attempt),
                    "run_json_sha256": sha256(attempt / "run.json"),
                    "native_image_path": item["native_path"],
                    "prediction_key": item["key"],
                },
                native,
            )
            rec.update(metrics)
            write(case / "graph.json", rec)
            write(
                case / "result.json",
                {
                    "native_sha256": item["native_sha256"],
                    "key": item["key"],
                    "protocol": protocol(cfg),
                    "status": "ok",
                    "execution_attempt": str(attempt),
                    **metrics,
                },
            )
            finish(case)
            pred = prediction(
                case / "graph.json", case / "receipt.json", case / "native.npz", case / "decoded.json", "unified"
            )
            accept(images, item["native_sha256"], pred)
            generated += 1
            save_index(out, cfg, rows, images)
            tracking.log(
                {
                    "new_images_completed": generated,
                    "cached_images": len(images) - len(pending) + generated,
                    "parse_success": 1,
                    "seconds": metrics["elapsed_seconds"],
                }
            )
            print(
                json.dumps(
                    {"new_image": generated, "requested_new": len(pending), "uses": item["uses"], "status": "ok"}
                ),
                flush=True,
            )
            case = None
    except (Exception, KeyboardInterrupt) as exc:
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "oom": "out of memory" in str(exc).lower(),
            "prediction_attempt": str(case) if case else None,
        }
        if case is not None:
            write(case / "failure.json", error)
    finally:
        if backend is not None:
            try:
                runtime = backend.close()
            except Exception as exc:
                error = error or {"type": type(exc).__name__, "message": str(exc), "stage": "CUDA_profile"}
        summary = {
            "requested_pairs": len(rows),
            "unique_images": len(images),
            "reused_images": len(images) - len(pending),
            "new_images_requested": len(pending),
            "new_images_completed": generated,
            "new_images_attempted": attempted,
            "parse_success_on_attempted_images": generated / attempted if attempted else None,
            "successful_images": sum(i["status"] == "ok" for i in images.values()),
            "error": error,
            "seconds_including_load_inference": time.perf_counter() - started,
            **runtime,
            "quality_metrics": None,
            "quality_status": "requires_graph_GT_or_manual_review",
            "attributes": "unsupported_no_attribute_head",
            "automatic_next_run": False,
        }
        write(attempt / "execution.json", summary)
        atomic(out / "last-attempt.json", {"path": str(attempt), **summary})
        if tracking is not None:
            tracking.finish(exit_code=0 if error is None else 1)
        refresh(out, cfg, rows, images)
    print(json.dumps({"output": str(out), **summary}), flush=True)
    return 2 if error else 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["preflight", "collect", "infer", "serve"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--no-import", action="store_true", help="For a fresh 300-image reproduction in a separate output")
    p.add_argument("--max-new-images", type=int, default=100)
    p.add_argument("--port", type=int, default=8009)
    args = p.parse_args()
    code = identity()
    cfg, rows = inputs()
    from .backend import preflight

    if args.command in ("preflight", "infer"):
        preflight()  # hashes + wheel/provider capabilities; no model session or GPU work
    out = workspace(args.output, cfg, code)
    with lock(out / ".state.lock"):
        images = collect(out, cfg, rows, use_legacy=not args.no_import)
        refresh(out, cfg, rows, images)
        pending = [i for i in images.values() if i["status"] != "ok"]
        print(
            json.dumps(
                {
                    "pairs": len(rows),
                    "unique_images": len(images),
                    "cached_images": len(images) - len(pending),
                    "pending_images": len(pending),
                    "pending_roles": dict(Counter(u["role"] for i in pending for u in i["uses"])),
                }
            ),
            flush=True,
        )
        if args.command == "infer" and pending:
            if len(pending) > args.max_new_images:
                raise ValueError("new_image_budget_exceeded_before_GPU")

            def terminate(_sig, _frame):
                raise KeyboardInterrupt("SIGTERM; completed predictions remain resumable")

            signal.signal(signal.SIGTERM, terminate)
            with lock(GPU_LOCK):
                return infer_pending(out, cfg, rows, images, code, args.max_new_images)
    if args.command == "serve":
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(out))
        with http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
            print(f"http://localhost:{args.port}/comparison.html", flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
    return 0
