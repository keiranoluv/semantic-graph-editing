"""Task 3 CLI: author, freeze intent, smoke, infer, review, report. No training."""

import argparse
import copy
import fcntl
import importlib.metadata
import json
import os
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

from tools.level3_image_graph_v1.core import DEFAULT_LOCKED, ROOT, digest, read, sha256, write
from tools.level3_sgg_pretrained_v2.assets import verify as verify_sgg
from tools.level3_sgg_pretrained_v2.runner import verified_bundle

from .assets import prepare, verify
from .core import (
    ASSETS,
    CONFIG,
    DRAFTS,
    RUNS,
    approved_pack,
    code_identity,
    finish,
    output_dir,
    sources,
    validate_tasks,
)
from .scoring import canonical_source, correlate, score
from .viewer import authoring, outcome_review


def context(cfg):
    return {
        "config_sha256": sha256(CONFIG),
        "source_receipt_sha256": cfg["source_receipt_sha256"],
        "draft_config_sha256": sha256(DRAFTS),
    }


def author(args):
    cfg, records = sources()
    pack = validate_tasks(read(DRAFTS), records)
    out = output_dir(args.output)
    authoring(pack, records, out, context(cfg))
    finish(out, "authoring")
    print(out / "authoring.html")


def compile_tasks(args):
    cfg, records = sources()
    submitted = read(args.submission)
    draft = read(DRAFTS)
    if (
        submitted["schema"] != "level3-task3-authoring-v1"
        or submitted["context"] != context(cfg)
        or submitted["draft_sha256"] != digest(draft)
    ):
        raise ValueError("stale_authoring_submission")
    pack = validate_tasks(submitted["pack"], records, approved=True)
    expected = {t["task_id"]: t for t in draft["tasks"]}
    actual = {t["task_id"]: t for t in pack["tasks"]}
    if set(actual) != set(expected):
        raise ValueError("must_review_exact_ten_tasks")
    for tid, t in actual.items():
        if any(t[k] != expected[tid][k] for k in ("image_id", "source_sha256")):
            raise ValueError("authoring_image_identity_changed")
    # Canonical ordering makes resume/reproduction independent of UI sort order.
    pack["tasks"] = [actual[t["task_id"]] for t in draft["tasks"]]
    out = output_dir(args.output)
    write(out / "tasks.json", pack)
    write(out / "authoring-submission.json", submitted)
    write(
        out / "provenance.json",
        {"context": context(cfg), "submission_sha256": sha256(args.submission), "code": code_identity()},
    )
    finish(out, "approved_tasks")
    print(out)


def run_pipeline(args):
    cfg, originals = sources()
    verify()
    verify_sgg()
    smoke = args.command == "smoke"
    pack = validate_tasks(read(DRAFTS), originals) if smoke else approved_pack(args.tasks, originals)
    tasks = copy.deepcopy(pack["tasks"][:1] if smoke else pack["tasks"])
    ecfg = copy.deepcopy(cfg["editor"])
    if smoke:
        ecfg.update({k: cfg["smoke"][k] for k in ("steps", "max_side")})
    seed = cfg["smoke"]["seed"] if smoke else args.seed
    if seed not in cfg["seeds"]:
        raise ValueError("seed_outside_protocol")
    import torch

    from .backends import LPIPS, ClipScore, Editor, Extractor, image_seed, release, resize_source

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.device == "cuda":
        if not torch.cuda.is_available() or "sm_120" not in torch.cuda.get_arch_list():
            raise RuntimeError("CUDA128_Blackwell_runtime_required")
    out = output_dir(args.output)
    for name in ("images", "native", "graphs"):
        (out / name).mkdir()
    provenance = {
        "schema": "level3-task3-run-v1",
        "mode": args.command,
        "evaluation_eligible": not smoke,
        "config": cfg,
        "effective_editor_config": ecfg,
        "seed": seed,
        "task_pack_sha256": digest(pack),
        "approved_tasks_receipt_sha256": None if smoke else sha256(args.tasks / "receipt.json"),
        "code": code_identity(),
        "asset_lock_sha256": sha256(ASSETS),
        "device": args.device,
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "stage_order": ["editor", "REACT++", "CLIP", "LPIPS"],
        "automatic_queue": False,
        "training": False,
    }
    write(out / "run.json", provenance)
    write(out / "tasks.json", {"schema": pack["schema"], "tasks": tasks})
    write(out / "environment.json", {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()})
    tracking = None
    results = []
    stage_times = {}
    started = time.perf_counter()
    try:
        if not smoke:
            import wandb

            if os.environ.get("WANDB_MODE", "online") != "online":
                raise ValueError("main_pilot_requires_online_WandB")
            tracking = wandb.init(
                project="level3-task3-v1", name=out.name, config=provenance, dir=str(out), mode="online"
            )
            write(out / "wandb-run.json", {"id": tracking.id, "url": tracking.url})
        source_paths = {}
        for cid in dict.fromkeys(t["image_id"] for t in tasks):
            path = out / "images" / (cid + "-source.png")
            resize_source(DEFAULT_LOCKED / originals[cid]["image"]["image_path"], ecfg["max_side"]).save(path)
            source_paths[cid] = path
        # One editor pipeline at a time, all edits independently start from source.
        t0 = time.perf_counter()
        editor = Editor(args.device, ecfg)
        for task in tasks:
            cid = task["image_id"]
            case = f"{task['task_id']}-seed{seed}"
            path = out / "images" / (case + ".png")
            result = {
                "case_id": case,
                "task_id": task["task_id"],
                "image_id": cid,
                "seed": seed,
                "instruction_en": task["instruction_en"],
                "target_caption_en": task["target_caption_en"],
                "operation": task["operation"],
                "source_path": str(source_paths[cid]),
                "source_sha256": sha256(source_paths[cid]),
                "status": "pending",
                "evaluation_eligible": not smoke,
            }
            with Image.open(source_paths[cid]) as im:
                generated, info = editor.edit(
                    im.convert("RGB"), task["instruction_en"], image_seed(seed, task["task_id"])
                )
            generated.save(path)
            result.update(target_path=str(path), target_sha256=sha256(path), editor=info)
            result["status"] = "safety_filtered" if info["safety_checker_flagged"] else "edited"
            results.append(result)
            write(out / "native" / (case + "-editor.json"), info)
            print(json.dumps({"case": case, "stage": "editor", "status": result["status"]}), flush=True)
        del editor
        release()
        stage_times["editor_seconds"] = time.perf_counter() - t0
        # Re-extract resized source as control: avoids mixing original resolution with editor output resolution.
        t0 = time.perf_counter()
        extractor = Extractor(args.device, out, read(ROOT / "configs/level3/sgg-pretrained-v2.json")["react"])
        resized_sources = {}
        control_evidence = {}
        for cid, path in source_paths.items():
            raw = extractor.extract(path, cid + "-resize", out / "native" / (cid + "-source.npz"), provenance)
            control, diagnostic = canonical_source(originals[cid], raw, cfg["scoring"])
            resized_sources[cid] = control
            control_evidence[cid] = diagnostic
            write(out / "graphs" / (cid + "-source-native.json"), raw)
            write(out / "graphs" / (cid + "-source.json"), control)
        write(out / "resize-control.json", control_evidence)
        task_by_id = {t["task_id"]: t for t in tasks}
        for result in results:
            if result["status"] != "edited":
                continue
            case = result["case_id"]
            cid = result["image_id"]
            try:
                target = extractor.extract(
                    Path(result["target_path"]), case, out / "native" / (case + ".npz"), provenance
                )
                write(out / "graphs" / (case + "-target.json"), target)
                result["graph_scores"] = score(
                    resized_sources[cid], target, task_by_id[result["task_id"]], cfg["scoring"]
                )
                result["graph_consistency"] = result["graph_scores"]["graph_consistency"]
                result["source_graph_path"] = str(out / "graphs" / (cid + "-source.json"))
                result["target_graph_path"] = str(out / "graphs" / (case + "-target.json"))
                result["status"] = "ok"
            except (ValueError, KeyError, IndexError) as error:
                result.update(status="graph_parse_error", error=repr(error), graph_consistency=None)
        write(out / "react-execution.json", extractor.close())
        del extractor
        release()
        stage_times["react_seconds"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        clip = ClipScore(args.device)
        for result in results:
            if result["status"] == "safety_filtered":
                continue
            try:
                with Image.open(result["target_path"]) as im:
                    result.update(clip.score(im.convert("RGB"), result["target_caption_en"]))
            except ValueError as error:
                result.update(clipscore=None, clip_error=repr(error))
        del clip
        release()
        stage_times["clip_seconds"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        lpips = LPIPS(args.device)
        for result in results:
            if result["status"] == "safety_filtered":
                continue
            with Image.open(result["source_path"]) as a, Image.open(result["target_path"]) as b:
                result.update(lpips.score(a, b))
        del lpips
        release()
        stage_times["lpips_seconds"] = time.perf_counter() - t0
        for result in results:
            for key in ("graph_consistency", "clipscore", "lpips", "negative_lpips"):
                result.setdefault(key, None)
            if tracking:
                tracking.log(
                    {
                        k: v
                        for k, v in {
                            "graph_consistency": result["graph_consistency"],
                            "clipscore": result["clipscore"],
                            "lpips": result["lpips"],
                            "run_success": int(result["status"] == "ok"),
                        }.items()
                        if v is not None
                    }
                )
        stage_times["total_seconds"] = time.perf_counter() - started
        write(
            out / "execution.json",
            {
                "stage_times": stage_times,
                "torch_cuda_initialized": torch.cuda.is_initialized(),
                "tasks_attempted": len(results),
                "successful": sum(r["status"] == "ok" for r in results),
                "quality_measured_by_humans": False,
            },
        )
        write(out / "results.json", results)
        outcome_review(results, out, smoke=smoke)
        if tracking:
            tracking.finish()
            tracking = None
        finish(out, args.command)
        print(json.dumps({"output": str(out), "tasks": len(results), "smoke": smoke}), flush=True)
        return 0 if all(r["status"] == "ok" for r in results) else 2
    except Exception as error:
        if tracking:
            tracking.finish(exit_code=1)
        write(out / "failure.json", {"error": repr(error), "partial_results": results, "execution_complete": False})
        raise


def report(args):
    results = verified_bundle(args.run, "results.json")
    run = read(args.run / "run.json")
    if not run["evaluation_eligible"]:
        raise ValueError("smoke_is_not_benchmark_evidence")
    review = read(args.review)
    if review["schema"] != "level3-task3-outcome-review-v1" or review["results_sha256"] != digest(results):
        raise ValueError("stale_outcome_review")
    index = {r["case_id"]: r for r in results}
    labels = review["reviews"]
    if len(labels) != len(index) or {r["case_id"] for r in labels} != set(index):
        raise ValueError("review_must_cover_all_outputs")
    for r in labels:
        if r["result_sha256"] != digest(index[r["case_id"]]):
            raise ValueError("review_result_changed")
    metrics = correlate(results, labels)
    metrics.update(
        total_outputs=len(results),
        failed_outputs=sum(r["status"] != "ok" for r in results),
        run_receipt_sha256=sha256(args.run / "receipt.json"),
        review_sha256=sha256(args.review),
    )
    out = output_dir(args.output)
    write(out / "metrics.json", metrics)
    write(out / "submitted-review.json", review)
    finish(out, "report")
    print(json.dumps(metrics, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("prepare", "preflight"):
        sub.add_parser(name)
    q = sub.add_parser("author")
    q.add_argument("--output", type=Path, default=RUNS / "authoring-v1")
    q = sub.add_parser("compile")
    q.add_argument("--submission", type=Path, required=True)
    q.add_argument("--output", type=Path, default=RUNS / "tasks-approved-v1")
    for name in ("smoke", "infer"):
        q = sub.add_parser(name)
        q.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
        q.add_argument("--output", type=Path, required=True)
        if name == "infer":
            q.add_argument("--tasks", type=Path, default=RUNS / "tasks-approved-v1")
            q.add_argument("--seed", type=int, choices=(42, 123, 3407), default=42)
    q = sub.add_parser("report")
    q.add_argument("--run", type=Path, required=True)
    q.add_argument("--review", type=Path, required=True)
    q.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.command == "prepare":
        prepare()
        return 0
    if args.command == "preflight":
        cfg, records = sources()
        validate_tasks(read(DRAFTS), records)
        verify()
        verify_sgg()
        print(
            json.dumps(
                {
                    "status": "ok",
                    "source_images": len(records),
                    "draft_tasks": len(read(DRAFTS)["tasks"]),
                    "approved": False,
                }
            )
        )
        return 0
    if args.command == "author":
        author(args)
        return 0
    if args.command == "compile":
        compile_tasks(args)
        return 0
    if args.command == "report":
        report(args)
        return 0
    with (ROOT / "data/cache/level3-sgg-v1/inference.lock").open("a") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run_pipeline(args)
