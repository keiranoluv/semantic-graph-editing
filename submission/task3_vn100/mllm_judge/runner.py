"""One independent stage per invocation. Case records survive interrupted execution."""

import os
import uuid

from task3_vn100.qwen_unified.core import GPU_LOCK, lock

from task3_vn100.mllm_judge.backend import Judge
from task3_vn100.mllm_judge.protocol import RELEASE, RUN, atomic, committed, digest, load_bound, parse, read, save_bound, sha256


def protocol_header(cfg, code, main, smoke):
    return {"config": digest(cfg), "code": code, "vn100_manifest": digest(main), "smoke_manifest": digest(smoke)}


def smoke_dir(header):
    return RUN.parent / "smoke" / digest(header)[:16]


def case_path(folder, case):
    return folder / "cases" / (case["task_id"] + ".json")


def saved_case(folder, case, header):
    path = case_path(folder, case)
    row = load_bound(path)
    if (
        row["task_id"] != case["task_id"]
        or row["input_sha256"] != digest(case)
        or row["protocol"] != digest(header)
        or row["scores"] != parse(row["raw"])
    ):
        raise ValueError(f"Stale case output: {path}")
    return row


def collect(folder, cases, header):
    paths = {case_path(folder, c) for c in cases}
    if set((folder / "cases").glob("*.json")) != paths:
        raise ValueError(f"Expected exactly {len(cases)} saved cases: {folder}")
    return [saved_case(folder, c, header) for c in cases]


def summary(rows):
    return {
        "attempted": len(rows),
        "valid": sum(r["scores"]["status"] == "ok" for r in rows),
        "case_digests": {r["task_id"]: digest(r) for r in rows},
        "max_score_peak_mib": {k: max(r["peak"][k] for r in rows) for k in ("allocated_mib", "reserved_mib")},
        "max_load_peak_mib": {k: max(r["load"]["peak"][k] for r in rows) for k in ("allocated_mib", "reserved_mib")},
    }


def verified(folder, cases, header):
    if load_bound(folder / "protocol.json") != header:
        raise ValueError("Changed stage protocol")
    rows = collect(folder, cases, header)
    receipt = load_bound(folder / "complete.json")
    if receipt["header"] != header or receipt["summary"] != summary(rows):
        raise ValueError("Completion receipt no longer matches the stage")
    return rows


def release_record(header, smoke):
    folder = smoke_dir(header)
    rows = verified(folder, smoke, header)
    if len(rows) != 5 or any(r["scores"]["status"] != "ok" for r in rows) or (folder / "oom.json").exists():
        raise ValueError("Smoke requires 5/5 valid cases and no OOM")
    return {
        "schema": "level3-mllm-judge-release-v1",
        "protocol": header,
        "smoke_summary": summary(rows),
        "smoke_receipt_sha256": sha256(folder / "complete.json"),
        "review": "Reviewed all five raw responses for one readable evidence sentence and correct image roles.",
    }


def check_release(header, smoke):
    if not RELEASE.is_file():
        raise ValueError("Smoke and committed CPU freeze are required before scoring")
    committed([RELEASE])
    if read(RELEASE) != release_record(header, smoke):
        raise ValueError("Release mismatch; use a new version for protocol changes")


def tracking(folder, header, cfg):
    import wandb

    if os.environ.get("WANDB_MODE", "online") != "online":
        raise ValueError("The main run requires online W&B")
    id_path = folder / "wandb-id.json"
    if not id_path.exists():
        atomic(id_path, {"id": uuid.uuid4().hex})
    run = wandb.init(
        project="level3-mllm-judge-v1",
        name="vn100-seed42",
        mode="online",
        dir=str(folder),
        id=read(id_path)["id"],
        resume="allow",
        save_code=False,
        settings=wandb.Settings(disable_git=True, console="off"),
        config={
            "protocol_sha256": digest(header),
            **{
                k: cfg[k]
                for k in (
                    "model",
                    "revision",
                    "seed",
                    "batch_size",
                    "dtype",
                    "attention",
                    "do_sample",
                    "num_beams",
                    "max_new_tokens",
                    "long_edge",
                    "aggregation",
                )
            },
        },
    )
    if run is None or run.offline:
        raise RuntimeError("W&B online initialization required")
    return run


def execute(folder, cases, header, cfg, *, main, factory=Judge):
    with lock(folder / ".lock"):
        if (folder / "complete.json").exists():
            return verified(folder, cases, header)
        header_path = folder / "protocol.json"
        if header_path.exists() and load_bound(header_path) != header:
            raise ValueError("Refusing to resume with a different protocol")
        save_bound(header_path, header)
        pending = []
        valid = 0
        for case in cases:
            if case_path(folder, case).exists():
                valid += saved_case(folder, case, header)["scores"]["status"] == "ok"
            else:
                pending.append(case)
        run = tracking(folder, header, cfg) if main else None
        try:
            if pending:
                with lock(GPU_LOCK):
                    judge = factory(cfg)
                    for index, case in enumerate(pending, start=len(cases) - len(pending) + 1):
                        result = judge(case)
                        scores = parse(result["raw"])
                        row = {
                            "task_id": case["task_id"],
                            "input_sha256": digest(case),
                            "protocol": digest(header),
                            "load": judge.load,
                            **result,
                            "scores": scores,
                        }
                        save_bound(case_path(folder, case), row)
                        valid += scores["status"] == "ok"
                        print(f"{index}/{len(cases)} {case['task_id']} {scores['status']}", flush=True)
                        if run:
                            run.log(
                                {
                                    "completed": index,
                                    "valid": valid,
                                    "tokens": result["tokens"],
                                    "seconds": result["seconds"],
                                    "score_peak": result["peak"],
                                    "load_peak": judge.load["peak"],
                                }
                            )
            rows = collect(folder, cases, header)
            stats = summary(rows)
            if run:
                run.summary.update({k: v for k, v in stats.items() if k != "case_digests"})
                run.finish()
            save_bound(
                folder / "complete.json",
                {
                    "header": header,
                    "summary": stats,
                    "wandb_id": read(folder / "wandb-id.json")["id"] if main else None,
                },
            )
            return rows
        except BaseException as exc:
            import torch

            if isinstance(exc, torch.OutOfMemoryError):
                atomic(folder / "oom.json", {"error": str(exc)})
            if run:
                run.finish(exit_code=1)
            raise
