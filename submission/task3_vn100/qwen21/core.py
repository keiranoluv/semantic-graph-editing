"""CPU-only contracts for paired inputs, frozen assets and bounded execution."""

import subprocess
from pathlib import Path

from task3_vn100.image_graph.core import ROOT, digest, read, sha256, write

CONFIG = ROOT / "configs/level3/qwen21-pilot-v1.json"
LOCK = ROOT / "configs/level3/qwen21-assets-v1.json"
CACHE = ROOT / "data/cache/level3-qwen21-v1"
RUNS = ROOT / "runs/level3/qwen21-v1"


def bundle(path, name):
    path = Path(path).resolve()
    receipt = read(path / "receipt.json")
    if not receipt["execution_complete"] or name not in receipt["files_sha256"]:
        raise ValueError("incomplete_or_unbound_bundle")
    for f, checksum in receipt["files_sha256"].items():
        target = (path / f).resolve()
        if path not in target.parents or sha256(target) != checksum:
            raise ValueError("changed_artifact:" + f)
    return read(path / name)


def finish(path, success=True):
    write(
        path / "receipt.json",
        {
            "schema": "qwen21-receipt-v1",
            "execution_complete": success,
            "files_sha256": {str(p.relative_to(path)): sha256(p) for p in sorted(path.rglob("*")) if p.is_file()},
        },
    )


def output(path):
    path = Path(path).resolve()
    if RUNS.resolve() not in path.parents:
        raise ValueError("output_must_be_under_qwen21_runs")
    path.mkdir(parents=True, exist_ok=False)
    return path


def inputs():
    cfg = read(CONFIG)
    if (
        cfg["model"] != "Qwen/Qwen-Image-2.1"
        or cfg["seed"] != 42
        or cfg["num_inference_steps"] != 40
        or cfg["dtype"] != "bfloat16"
        or cfg["batch_size"] != 1
        or cfg["training"]
        or cfg["automatic_queue"]
    ):
        raise ValueError("invalid_fixed_protocol")
    base, approved = ROOT / cfg["baseline_run"], ROOT / cfg["approved_tasks"]
    for p, key in [(base, "baseline_receipt_sha256"), (approved, "approved_receipt_sha256")]:
        if sha256(p / "receipt.json") != cfg[key]:
            raise ValueError("parent_receipt_changed")
    rows = bundle(base, "results.json")
    tasks = bundle(approved, "tasks.json")["tasks"]
    index = {r["task_id"]: r for r in rows}
    task_index = {t["task_id"]: t for t in tasks}
    for tid in [cfg["smoke_task"], *cfg["pilot_tasks"]]:
        r, t = index[tid], task_index[tid]
        if r["status"] != "ok" or t["status"] != "approved" or not t["reviewer"].strip():
            raise ValueError("paired_approved_baseline_required")
        if r["instruction_en"] != t["instruction_en"] or r["image_id"] != t["image_id"]:
            raise ValueError("paired_prompt_changed")
        for tag in ("source", "target"):
            p = Path(r[tag + "_path"]).resolve()
            if base.resolve() not in p.parents or sha256(p) != r[tag + "_sha256"]:
                raise ValueError("paired_image_changed")
    if index[cfg["smoke_task"]]["image_id"] != "sidewalk-b1bb633496439f":
        raise ValueError("wrong_smoke_image")
    ids = [cfg["smoke_task"], *cfg["pilot_tasks"]]
    if len(cfg["pilot_tasks"]) > 5 or len(ids) != len(set(ids)):
        raise ValueError("pilot_limit_or_duplicate")
    return cfg, index


def select(cfg, index, mode, smoke_run=None):
    if mode == "smoke":
        return [index[cfg["smoke_task"]]]
    if mode != "pilot" or smoke_run is None:
        raise ValueError("successful_smoke_required")
    rows = bundle(smoke_run, "results.json")
    run = read(Path(smoke_run) / "run.json")
    if (
        run["mode"] != "smoke"
        or run["config_sha256"] != sha256(CONFIG)
        or run["asset_lock_sha256"] != sha256(LOCK)
        or run["device"] != "cuda"
        or len(rows) != 1
        or rows[0]["task_id"] != cfg["smoke_task"]
        or rows[0]["status"] != "ok"
    ):
        raise ValueError("successful_matching_GPU_smoke_required")
    return [index[tid] for tid in cfg["pilot_tasks"]]


def assets():
    lock = read(LOCK)
    if lock["repo_id"] != "Qwen/Qwen-Image-2.1" or lock["revision"] != read(CONFIG)["revision"]:
        raise ValueError("checkpoint_revision_mismatch")
    root = (CACHE / "model").resolve()
    for f in lock["files"]:
        p = (root / f["path"]).resolve()
        if root not in p.parents or p.stat().st_size != f["size"] or sha256(p) != f["sha256"]:
            raise ValueError("asset_changed:" + f["path"])
    return lock


def code():
    from common.provenance import source_hashes
    return {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "files_sha256": source_hashes(),
    }


def review_template(rows):
    return {
        "schema": "qwen21-manual-review-v1",
        "results_sha256": digest(rows),
        "reviews": [
            {
                "task_id": r["task_id"],
                "model": m,
                "status": "pending",
                "reviewer": "",
                "notes": "",
                "edit_fulfillment": None,
                "preservation": None,
            }
            for r in rows
            for m in ("ip2p", "qwen21")
        ],
    }
