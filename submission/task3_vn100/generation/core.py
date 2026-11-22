"""Mutable index over immutable per-case evidence; native images are never copied on import."""

import contextlib
import fcntl
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path

from common.paths import ROOT
from task3_vn100.image_graph import digest, read, sha256, write
from task3_vn100.qwen_assets import LOCK, bundle

CONFIG = ROOT / "configs/level3/qwen-unified-v1.json"

RUNS = ROOT / "runs/level3/qwen200"

GPU_LOCK = ROOT / "data/cache/level3-sgg-v1/inference.lock"


def atomic(path, value):
    """Readers see either the old snapshot or the whole new JSON document."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".update-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


@contextlib.contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another_process_holds_lock:" + str(path)) from exc
        yield


def inputs():
    cfg = read(CONFIG)
    manifest = ROOT / cfg["manifest"]
    if sha256(manifest) != cfg["manifest_sha256"] or sha256(LOCK) != cfg["asset_lock_sha256"]:
        raise ValueError("frozen_manifest_or_asset_lock_changed")
    tasks = read(manifest)["tasks"]
    if (
        len(tasks) != 200
        or len({t["task_id"] for t in tasks}) != 200
        or Counter(Counter(t["image_id"] for t in tasks).values()) != {2: 100}
        or Counter(t["dataset"] for t in tasks) != {"sidewalk": 100, "travel": 100}
        or sum(t["operation"]["kind"] == "attribute" for t in tasks) != 10
    ):
        raise ValueError("fixed_200_request_cohort_required")
    gen = cfg["generation"]
    if (
        cfg["expected_requests"] != 200
        or cfg["unique_sources"] != 100
        or cfg["batch_size"] != 1
        or cfg["training"]
        or cfg["automatic_next_run"]
        or cfg["quality_retries"]
        or gen["model"] != "Qwen/Qwen-Image-2.1"
        or gen["seed"] != 42
        or gen["num_inference_steps"] != 40
        or gen["dtype"] != "bfloat16"
        or gen["memory_mode"] != "model_cpu_offload"
        or gen["revision"] != read(LOCK)["revision"]
        or gen["output_resolution"] != 1024
        or gen["true_cfg_scale"] != 1.0
        or gen["use_kv_cache"] is not True
    ):
        raise ValueError("fixed_generation_protocol_changed")
    checked = {}
    for task in tasks:
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", task["task_id"]):
            raise ValueError("unsafe_task_id")
        p = Path(task["source_path"]).resolve()
        if (ROOT / "data/raw/level3/qwen21-batch100-v1").resolve() not in p.parents:
            raise ValueError("original_source_required")
        if p not in checked:
            checked[p] = sha256(p)
        if checked[p] != task["sha256"]:
            raise ValueError("original_source_changed")
    return cfg, tasks


def protocol(cfg):
    return {k: cfg[k] for k in ("manifest_sha256", "generation", "batch_size", "asset_lock_sha256")}


def identity():
    from common.provenance import revision, source_hashes

    return {"commit": revision(), "files_sha256": source_hashes()}


def workspace(path, cfg, code):
    path = Path(path).resolve()
    if RUNS.resolve() not in path.parents:
        raise ValueError("output_outside_unified_run_root")
    path.mkdir(parents=True, exist_ok=True)
    header = path / "run.json"
    if header.exists():
        if read(header)["protocol"] != protocol(cfg):
            raise ValueError("resume_protocol_changed")
    else:
        if any(path.iterdir()):
            raise ValueError("refuse_nonempty_unrecognized_output")
        write(header, {"schema": "qwen-unified-run-v1", "protocol": protocol(cfg), "code": code})
    return path


def bind(task, result):
    return digest(
        {
            "task_id": task["task_id"],
            "prompt": task["instruction_en"],
            "original_source_sha256": task["sha256"],
            "source_sha256": result["source_sha256"],
            "target_sha256": result["target"]["sha256"],
        }
    )


def load_case(case, task):
    case = Path(case).resolve()
    row = bundle(case, "result.json")
    if (
        Path(row["case_dir"]).resolve() != case
        or row["status"] != "ok"
        or row["task_id"] != task["task_id"]
        or row["image_id"] != task["image_id"]
        or row["prompt"] != task["instruction_en"]
        or row["native_source_sha256"] != task["sha256"]
        or row["operation"] != task["operation"]
        or row["seed"] != 42
        or row["dtype"] != "bfloat16"
        or row["completed_steps"] != 40
        or row["steps_requested"] != 40
    ):
        raise ValueError("case_does_not_match_request:" + task["task_id"])
    for role, expected in [("source", row["source_sha256"]), ("target", row["target"]["sha256"])]:
        p = Path(row[role + "_path"]).resolve()
        if case not in p.parents or sha256(p) != expected:
            raise ValueError("case_image_changed_or_escaped")
    return {
        "result": row,
        "case_dir": str(case),
        "case_receipt_sha256": sha256(case / "receipt.json"),
        "binding": bind(task, row),
    }


def validate_import(run, cfg):
    run = Path(run).resolve()
    if (ROOT / "runs/level3").resolve() not in run.parents:
        raise ValueError("import_run_outside_level3")
    old = read(run / "run.json")
    if old.get("schema") == "qwen-unified-run-v1":
        if old["protocol"] != protocol(cfg):
            raise ValueError("import_protocol_changed")
    elif (
        old["manifest_sha256"] not in cfg["accepted_legacy_manifest_sha256"]
        or old["config"]["generation"] != cfg["generation"]
        or old["config"]["batch_size"] != 1
        or old["asset_lock_sha256"] != cfg["asset_lock_sha256"]
    ):
        raise ValueError("legacy_protocol_mismatch")
    return run, old


def collect(out, cfg, tasks, imports):
    """Only committed case receipts count, even while an old inference process is active."""
    expected = {t["task_id"]: t for t in tasks}
    accepted, warnings, legacy, candidates = {}, [], [], []
    previous_path = out / "index.json"
    if previous_path.exists():
        previous = read(previous_path)
        if previous["protocol"] != protocol(cfg):
            raise ValueError("index_protocol_changed")
        legacy = previous["legacy_reviews"]
        for entry in previous["entries"]:
            if entry["status"] == "ok":
                case = entry["case_dir"]
                if sha256(Path(case) / "receipt.json") != entry["case_receipt_sha256"]:
                    raise ValueError("imported_receipt_changed")
                candidates.append(Path(case))
    for path in [*imports, out]:
        path = Path(path).resolve()
        if not (path / "run.json").exists():
            warnings.append("import_not_started:" + str(path))
            continue
        run, meta = validate_import(path, cfg)
        candidates += [p.parent for p in run.glob("cases/*/receipt.json")]
        candidates += [p.parent for p in run.glob("cases/*/attempt-*/receipt.json")]
        # Legacy resumed runs can point to case receipts in earlier runs.
        if meta.get("schema") != "qwen-unified-run-v1" and (run / "receipt.json").exists():
            try:
                sealed = read(run / "receipt.json")["execution_complete"]
            except json.JSONDecodeError:
                warnings.append("run_receipt_not_sealed:" + str(run))
                sealed = False
            if sealed:
                results = bundle(run, "results.json")
                candidates += [Path(r["case_dir"]) for r in results if r["status"] == "ok"]
                legacy.append({"results_sha256": digest(results), "task_ids": [r["task_id"] for r in results]})
    for case in sorted(set(p.resolve() for p in candidates)):
        if (ROOT / "runs/level3").resolve() not in case.parents:
            raise ValueError("case_outside_level3")
        try:
            receipt = read(case / "receipt.json")
        except json.JSONDecodeError:
            warnings.append("case_receipt_not_sealed:" + str(case))
            continue
        if not receipt["execution_complete"]:
            continue
        row = read(case / "result.json")
        if row["task_id"] not in expected:
            raise ValueError("unknown_import_task")
        entry = load_case(case, expected[row["task_id"]])
        old = accepted.get(row["task_id"])
        if old and old["binding"] != entry["binding"]:
            raise ValueError("conflicting_successful_outputs:" + row["task_id"])
        accepted.setdefault(row["task_id"], entry)
    entries = []
    for i, task in enumerate(tasks, 1):
        entry = {"task_id": task["task_id"], "index": i, "image_id": task["image_id"], "status": "pending"}
        if task["task_id"] in accepted:
            entry.update(status="ok", **accepted[task["task_id"]])
        entries.append(entry)
    state = {
        "schema": "qwen-unified-index-v1",
        "protocol": protocol(cfg),
        "entries": entries,
        "successful": len(accepted),
        "requested": len(tasks),
        "warnings": sorted(set(warnings)),
        "legacy_reviews": list({g["results_sha256"]: g for g in legacy}.values()),
        "execution_complete": len(accepted) == len(tasks),
    }
    save_state(out, state)
    return state


def save_state(out, state):
    atomic(out / "index.json", state)
    # Stable single destination. Originals and sealed per-case records remain immutable.
    atomic(
        out / "results.json",
        [
            dict(e["result"], provenance={"case_receipt_sha256": e["case_receipt_sha256"]})
            if e["status"] == "ok"
            else {"task_id": e["task_id"], "status": "not_run"}
            for e in state["entries"]
        ],
    )
