import math
from pathlib import Path

from common.paths import ROOT
from task3_vn100.generation.core import atomic, load_case
from task3_vn100.image_graph import digest, read, sha256, validate_record

CONFIG = ROOT / "configs/level3/evaluation200-v1.json"

DEFAULT_OUTPUT = ROOT / "runs/level3/evaluation-v1/seed42"


def identity():
    from common.provenance import revision, source_hashes

    return {"commit": revision(), "files_sha256": source_hashes()}


def inputs():
    """Read-only validation. Never collect/rewrite a generation or graph index."""
    cfg = read(CONFIG)
    for path, expected in cfg["files_sha256"].items():
        if sha256(ROOT / path) != expected:
            raise ValueError("frozen_evaluation_input_changed:" + path)
    tasks = read(ROOT / cfg["cohort"])["tasks"]
    authored = read(ROOT / cfg["authoring"])
    qwen = read(ROOT / cfg["qwen_run"] / "index.json")
    sgg = read(ROOT / cfg["sgg_run"] / "index.json")
    pairs = read(ROOT / cfg["sgg_run"] / "results.json")
    if not qwen["execution_complete"] or not sgg["execution_complete"]:
        raise ValueError("incomplete_image_or_graph_run")
    ids = [t["task_id"] for t in tasks]
    if len(ids) != 200 or len(set(ids)) != 200:
        raise ValueError("expected_200_unique_tasks")
    for rows in (authored["tasks"], qwen["entries"], pairs):
        if [r["task_id"] for r in rows] != ids:
            raise ValueError("task_order_or_membership_changed")
    if authored["cohort_sha256"] != sha256(ROOT / cfg["cohort"]):
        raise ValueError("authoring_cohort_mismatch")
    records = []
    for task, author, entry, pair in zip(tasks, authored["tasks"], qwen["entries"], pairs, strict=True):
        case = load_case(entry["case_dir"], task)
        if (
            entry["status"] != "ok"
            or case["binding"] != entry["binding"]
            or case["case_receipt_sha256"] != entry["case_receipt_sha256"]
        ):
            raise ValueError("image_receipt_or_binding_changed")
        if author["author"] != "assistant" or author["human_ground_truth"] is not False:
            raise ValueError("authoring_is_not_human_outcome_GT")
        binding = author["grounding"]
        region = binding["region"]
        if (
            len(region) != 4
            or not all(math.isfinite(x) and 0 <= x <= 1 for x in region)
            or region[0] >= region[2]
            or region[1] >= region[3]
        ):
            raise ValueError("invalid_source_region")
        rec = {}
        for role in ("source", "target"):
            item = pair["roles"][role]
            if item["status"] != "ok" or sha256(item["graph_path"]) != item["graph_sha256"]:
                raise ValueError("graph_artifact_changed")
            if item["native_image_path"] != case["result"][role + "_path"]:
                raise ValueError("graph_image_pair_mismatch")
            rec[role] = read(item["graph_path"])
            validate_record(rec[role])
            if rec[role]["pair_task_id"] != task["task_id"] or rec[role]["image_role"] != role:
                raise ValueError("graph_role_mismatch")
        if (
            author["source_graph_sha256"] != pair["roles"]["source"]["graph_sha256"]
            or author["source_image_sha256"] != pair["roles"]["source"]["native_sha256"]
        ):
            raise ValueError("stale_source_grounding")
        records.append(
            {"task": task, "author": author, "pair": pair, "binding": entry["binding"], "image": case["result"], **rec}
        )
    return cfg, records


def workspace(output, cfg, code):
    out = Path(output).resolve()
    if (ROOT / "runs/level3/evaluation-v1").resolve() not in out.parents:
        raise ValueError("output_outside_evaluation_runs")
    binding = {"config_sha256": sha256(CONFIG), "protocol": cfg, "code_files_sha256": code["files_sha256"]}
    if (out / "run.json").exists():
        if read(out / "run.json")["binding"] != binding:
            raise ValueError("resume_protocol_or_code_changed_use_new_OUTPUT")
    else:
        if out.exists() and any(out.iterdir()):
            raise ValueError("refuse_unrecognized_nonempty_output")
        atomic(out / "run.json", {"schema": "level3-evaluation-run-v1", "binding": binding, "code": code})
    return out


def case_header(record):
    task = record["task"]
    return {
        "task_id": task["task_id"],
        "image_id": task["image_id"],
        "dataset": task["dataset"],
        "operation": task["operation"]["kind"],
        "binding": record["binding"],
    }


def save_stage(path, value):
    atomic(path, value)
    atomic(Path(path).with_suffix(".receipt.json"), {"sha256": sha256(path)})


def read_stage(path):
    path = Path(path)
    if sha256(path) != read(path.with_suffix(".receipt.json"))["sha256"]:
        raise ValueError("evaluation_stage_artifact_changed:" + str(path))
    return read(path)


def stage_binding(cfg, records):
    return digest({"config": cfg, "cases": [case_header(r) for r in records]})


def validate_stage(pack, records, binding):
    if pack.get("evaluation_eligible") is not True or pack.get("stage_binding") != binding:
        raise ValueError("wrong_stage_protocol_or_smoke_not_eligible")
    if [r["task_id"] for r in pack["rows"]] != [r["task"]["task_id"] for r in records]:
        raise ValueError("stage_task_membership_changed")
    for row, record in zip(pack["rows"], records, strict=True):
        if any(row.get(k) != v for k, v in case_header(record).items()):
            raise ValueError("stage_case_binding_mismatch")
