"""Commit-gated CPU graph scoring, followed by a separate assistant review view."""

import argparse
import importlib.metadata
import json
import os
import time
from collections import Counter
from pathlib import Path

from tools.level3_image_graph_v1.core import read, sha256, write
from tools.level3_qwen21_v1.core import bundle, finish
from tools.level3_shared.receipts import finish_run

from .core import CONFIG, identity, inputs, new_output, validate_assistant_review
from .scoring import score
from .viewer import render


def summarize(results):
    scored = [r for r in results if r["scores"]["graph_consistency"] is not None]
    means = {}
    for field in ("edit_score", "preservation_score", "graph_consistency"):
        values = [r["scores"][field] for r in results if r["scores"][field] is not None]
        means[field] = {"available_n": len(values), "mean": sum(values) / len(values) if values else None}
    return {
        "pairs": len(results),
        "status_counts": dict(Counter(r["scores"]["status"] for r in results)),
        "graph_score_coverage": len(scored) / len(results) if results else None,
        "abstention_reasons": dict(Counter(reason for r in results for reason in r["scores"]["reason_codes"])),
        "graph_evidence_means_not_image_accuracy": means,
        "human_labels_used": 0,
        "object_or_relation_GT_F1": None,
        "human_spearman": None,
        "CLIPScore": None,
        "LPIPS": None,
        "interpretation": "development_graph_heuristic; not pass_rate, human_labels, or model_accuracy",
    }


def run(output):
    cfg, records = inputs()
    code = identity()  # before any full-cohort score computation
    if os.environ.get("WANDB_MODE", "online") != "online":
        raise ValueError("online_WandB_required")
    import wandb

    out = new_output(output)
    write(
        out / "run.json",
        {"config": cfg, "config_sha256": sha256(CONFIG), "code": code, "device": "cpu", "GPU_inference": False},
    )
    write(out / "environment.json", {k: importlib.metadata.version(k) for k in ("numpy", "scipy", "Pillow", "wandb")})
    rows = []
    tracking = None
    started = time.perf_counter()
    try:
        tracking = wandb.init(
            project=cfg["wandb_project"],
            name=out.name,
            dir=str(out),
            config={"config": cfg, "code": code},
            mode="online",
        )
        write(out / "wandb.json", {"id": tracking.id, "url": tracking.url})
        for i, item in enumerate(records):
            t0 = time.perf_counter()
            scores = score(item["source"], item["target"], item["task"], item["grounding"], cfg)
            row = {
                "task_id": item["task"]["task_id"],
                "image_id": item["task"]["image_id"],
                "dataset": item["task"]["dataset"],
                "prompt": item["task"]["instruction_en"],
                "operation": item["task"]["operation"],
                "grounding": item["grounding"],
                "roles": item["pair"]["roles"],
                "scores": scores,
                "score_seconds": time.perf_counter() - t0,
            }
            rows.append(row)
            write(out / "cases" / (row["task_id"] + ".json"), row)
            log = {"case_index": i + 1, "scorable": int(scores["status"] == "scored")}
            log.update(
                {
                    k: scores[k]
                    for k in ("edit_score", "preservation_score", "graph_consistency")
                    if scores[k] is not None
                }
            )
            tracking.log(log)
        report = summarize(rows)
        report["seconds_including_tracking_start"] = time.perf_counter() - started
        write(out / "results.json", rows)
        write(out / "summary.json", report)
        render(rows, out)
        tracking.summary.update(
            {"pairs": len(rows), "coverage": report["graph_score_coverage"], "human_labels_used": 0}
        )
        tracking.finish()
        tracking = None
        finish_run(out, success=True)
        print(json.dumps({"output": str(out), **report}), flush=True)
    except Exception as exc:
        write(out / "failure.json", {"error": repr(exc), "computed_cases": len(rows)})
        if tracking:
            tracking.finish(exit_code=1)
        finish_run(out, success=False)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    p = sub.add_parser("score")
    p.add_argument("--output", required=True, type=Path)
    p = sub.add_parser("review")
    p.add_argument("--score-run", required=True, type=Path)
    p.add_argument("--assistant-review", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "preflight":
        cfg, records = inputs()
        print(
            json.dumps(
                {
                    "status": "ok",
                    "pairs": len(records),
                    "device": "cpu",
                    "assistant_groundings": len(records),
                    "human_labels_read": 0,
                }
            )
        )
    elif args.command == "score":
        run(args.output)
    else:
        rows = bundle(args.score_run, "results.json")
        pack = validate_assistant_review(read(args.assistant_review), rows, sha256(args.score_run / "receipt.json"))
        cfg = read(CONFIG)
        if set(r["task_id"] for r in pack["reviews"]) != set(cfg["assistant_review_task_ids"]):
            raise ValueError("review_selection_changed_after_scoring")
        out = new_output(args.output)
        write(out / "assistant-review.json", pack)
        write(
            out / "provenance.json",
            {
                "score_run": str(args.score_run.resolve()),
                "score_receipt_sha256": sha256(args.score_run / "receipt.json"),
                "review_sha256": sha256(args.assistant_review),
                "code": identity(),
            },
        )
        render(rows, out, pack)
        finish(out)
        print(out / "comparison.html")
    return 0
