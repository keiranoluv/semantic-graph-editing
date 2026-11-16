"""Verify image/model bindings before reusing immutable SGG predictions."""

import copy
import json
import subprocess
from collections import Counter
from pathlib import Path

from tools.level3_image_graph_v1.core import ROOT, digest, read, sha256, validate_record, write
from tools.level3_qwen21_v1.core import bundle, finish
from tools.level3_qwen_unified.core import atomic
from tools.level3_qwen_unified.core import inputs as qwen_inputs
from tools.level3_qwen_unified.core import load_case

CONFIG = ROOT / "configs/level3/qwen-sgg-unified-v1.json"
RUNS = ROOT / "runs/level3/qwen-sgg"
MODEL_FIELDS = (
    "backend",
    "asset_lock_sha256",
    "parent_sgg_config_sha256",
    "react",
    "normalization",
    "preprocess",
    "batch_size",
    "seed",
)


def protocol(cfg):
    return {k: cfg[k] for k in (*MODEL_FIELDS, "pipeline_files_sha256")}


def inputs():
    from tools.level3_sgg_pretrained_v2.assets import LOCK

    cfg = read(CONFIG)
    if (
        sha256(LOCK) != cfg["asset_lock_sha256"]
        or sha256(ROOT / "configs/level3/sgg-pretrained-v2.json") != cfg["parent_sgg_config_sha256"]
    ):
        raise ValueError("frozen_model_assets_or_parent_config_changed")
    _, tasks = qwen_inputs()
    root = ROOT / cfg["qwen_run"]
    if (
        sha256(root / "index.json") != cfg["qwen_index_sha256"]
        or sha256(ROOT / cfg["manifest"]) != cfg["manifest_sha256"]
    ):
        raise ValueError("frozen_200_image_pairs_changed")
    idx = read(root / "index.json")
    if not idx["execution_complete"] or idx["successful"] != 200 or len(idx["entries"]) != 200:
        raise ValueError("all_200_Qwen_results_required")
    if (
        cfg["expected_pairs"] != 200
        or cfg["expected_unique_images"] != 300
        or cfg["seed"] != 42
        or cfg["batch_size"] != 1
        or cfg["training"]
        or cfg["automatic_next_run"]
    ):
        raise ValueError("fixed_SGG_protocol_changed")
    for path, checksum in cfg["pipeline_files_sha256"].items():
        if sha256(ROOT / path) != checksum:
            raise ValueError("frozen_inference_dependency_changed:" + path)
    rows = []
    for entry, task in zip(idx["entries"], tasks, strict=True):
        if entry["task_id"] != task["task_id"] or entry["status"] != "ok":
            raise ValueError("image_pair_order_changed")
        case = load_case(entry["case_dir"], task)
        if case["case_receipt_sha256"] != entry["case_receipt_sha256"] or case["binding"] != entry["binding"]:
            raise ValueError("image_pair_receipt_changed")
        rows.append(case["result"])
    if Counter(Counter(r["image_id"] for r in rows).values()) != {2: 100}:
        raise ValueError("two_independent_requests_per_source_required")
    if len(inventory(rows, cfg)) != 300:
        raise ValueError("expected_300_distinct_native_images")
    return cfg, rows


def identity():
    from tools.level3_sgg_pretrained_v2.assets import LOCK

    files = list(Path(__file__).parent.glob("*.py")) + list(Path(__file__).parent.glob("*.html"))
    files += [
        CONFIG,
        ROOT / "scripts/run_level3_sgg.sh",
        ROOT / "requirements-level3-sgg-pretrained-v2.txt",
        LOCK,
        ROOT / read(CONFIG)["manifest"],
        ROOT / "configs/level3/qwen-unified-v1.json",
        ROOT / "configs/level3/sgg-pretrained-v2.json",
        ROOT / "tools/level3_qwen_unified/core.py",
        ROOT / "tools/level3_qwen_unified/runner.py",
        ROOT / "tools/level3_qwen21_v1/core.py",
        ROOT / "tools/level3_shared/qwen_runtime.py",
        ROOT / "tools/level3_shared/__init__.py",
        ROOT / "tools/level3_sgg_pretrained_v2/assets.py",
        ROOT / "configs/level3/qwen21-pilot-v1.json",
    ]
    files += [ROOT / p for p in read(CONFIG)["pipeline_files_sha256"]]
    files = sorted(set(files))
    if subprocess.check_output(["git", "status", "--porcelain", "--", *map(str, files)], text=True, cwd=ROOT):
        raise ValueError("commit_SGG_code_before_execution")
    return {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip(),
        "files_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in files},
    }


def workspace(out, cfg, code):
    out = Path(out).resolve()
    if RUNS.resolve() not in out.parents:
        raise ValueError("output_outside_unified_SGG_root")
    out.mkdir(parents=True, exist_ok=True)
    binding = {
        "protocol": protocol(cfg),
        "qwen_index_sha256": cfg["qwen_index_sha256"],
        "manifest_sha256": cfg["manifest_sha256"],
    }
    if (out / "run.json").exists():
        if read(out / "run.json")["binding"] != binding:
            raise ValueError("SGG_resume_protocol_changed")
    else:
        if any(out.iterdir()):
            raise ValueError("unknown_nonempty_SGG_output")
        write(out / "run.json", {"schema": "qwen-sgg-unified-run-v1", "binding": binding, "config": cfg, "code": code})
    return out


def image_hash(row, role):
    return row["source_sha256"] if role == "source" else row["target"]["sha256"]


def inventory(rows, cfg):
    images = {}
    sources = {}
    for row in rows:
        previous = sources.setdefault(row["image_id"], row["source_sha256"])
        if previous != row["source_sha256"]:
            raise ValueError("one_source_identity_has_different_images")
        for role in ("source", "target"):
            checksum = image_hash(row, role)
            item = images.setdefault(
                checksum,
                {
                    "native_sha256": checksum,
                    "native_path": row[role + "_path"],
                    "key": digest([checksum, protocol(cfg)]),
                    "status": "pending",
                    "uses": [],
                },
            )
            item["uses"].append({"task_id": row["task_id"], "role": role})
    return images


def check_prediction(pred, checksum):
    if sha256(pred["receipt_path"]) != pred["receipt_sha256"]:
        raise ValueError("prediction_receipt_changed")
    for path_key, hash_key in [
        ("record_path", "record_sha256"),
        ("normalized_path", "normalized_sha256"),
        ("raw_path", "raw_sha256"),
        ("decoded_path", "decoded_sha256"),
    ]:
        if sha256(pred[path_key]) != pred[hash_key]:
            raise ValueError("prediction_artifact_changed:" + path_key)
    rec = read(pred["record_path"])
    if (
        rec["status"] != "ok"
        or rec["image"]["native"]["sha256"] != checksum
        or rec["source_sha256"] != pred["normalized_sha256"]
    ):
        raise ValueError("prediction_image_binding_mismatch")
    validate_record(rec)
    return rec


def prediction(record, receipt, raw, decoded, kind):
    rec = read(record)
    norm = rec["image"]["image_path"]
    return {
        "record_path": str(record),
        "record_sha256": sha256(record),
        "receipt_path": str(receipt),
        "receipt_sha256": sha256(receipt),
        "normalized_path": norm,
        "normalized_sha256": sha256(norm),
        "raw_path": str(raw),
        "raw_sha256": sha256(raw),
        "decoded_path": str(decoded),
        "decoded_sha256": sha256(decoded),
        "origin": kind,
    }


def legacy_predictions(cfg, rows):
    old = ROOT / cfg["legacy_run"]
    if sha256(old / "receipt.json") != cfg["legacy_receipt_sha256"]:
        raise ValueError("legacy_SGG_receipt_changed")
    pairs = bundle(old, "results.json")
    run = read(old / "run.json")
    if run["config_sha256"] != cfg["legacy_config_sha256"] or any(run["config"][k] != cfg[k] for k in MODEL_FIELDS):
        raise ValueError("legacy_SGG_model_or_threshold_mismatch")
    if any(run["code"]["files_sha256"][p] != s for p, s in cfg["pipeline_files_sha256"].items()):
        raise ValueError("legacy_SGG_decoder_mismatch")
    expected = {r["task_id"]: r for r in rows}
    result = {}
    for pair in pairs:
        row = expected[pair["task_id"]]
        if pair["image_id"] != row["image_id"] or pair["prompt"] != row["prompt"]:
            raise ValueError("legacy_pair_mismatch")
        for role, item in pair["roles"].items():
            checksum = image_hash(row, role)
            rec_path = Path(item["graph_path"])
            rec = read(rec_path)
            if (
                old.resolve() not in rec_path.resolve().parents
                or item["status"] != "ok"
                or rec["pair_task_id"] != row["task_id"]
                or rec["image_role"] != role
                or rec["image"]["native"]["sha256"] != checksum
            ):
                raise ValueError("legacy_image_or_role_mismatch")
            if rec["image"]["policy"] != cfg["normalization"]:
                raise ValueError("legacy_normalization_mismatch")
            pred = prediction(
                rec_path,
                old / "receipt.json",
                rec_path.parent / rec["native_output"]["raw_npz"],
                rec_path.parent / (role + "-decoded.json"),
                "legacy",
            )
            check_prediction(pred, checksum)
            result[checksum] = pred
    return result


def accept(images, checksum, pred):
    if checksum not in images:
        raise ValueError("prediction_outside_current_cohort")
    check_prediction(pred, checksum)
    item = images[checksum]
    if item["status"] == "ok" and item["prediction"]["record_sha256"] != pred["record_sha256"]:
        raise ValueError("conflicting_successful_predictions")
    item.update(status="ok", prediction=pred)


def collect(out, cfg, rows, use_legacy=True):
    images = inventory(rows, cfg)
    prior_path = out / "index.json"
    if prior_path.exists():
        prior = read(prior_path)
        if prior["protocol"] != protocol(cfg):
            raise ValueError("cache_protocol_changed")
        for checksum, item in prior["images"].items():
            if item["status"] == "ok":
                accept(images, checksum, item["prediction"])
    if use_legacy:
        for checksum, pred in legacy_predictions(cfg, rows).items():
            # A deliberate fresh run may already contain a different valid prediction.
            if images[checksum]["status"] != "ok":
                accept(images, checksum, pred)
    for receipt in sorted(out.glob("predictions/*/attempt-*/receipt.json")):
        try:
            sealed = read(receipt)
        except json.JSONDecodeError:
            continue  # interrupted receipt is not a reusable success
        if not sealed["execution_complete"]:
            continue
        result = bundle(receipt.parent, "result.json")
        checksum = result["native_sha256"]
        if result["protocol"] != protocol(cfg) or result["key"] != images[checksum]["key"]:
            raise ValueError("native_prediction_protocol_mismatch")
        pred = prediction(
            receipt.parent / "graph.json",
            receipt,
            receipt.parent / "native.npz",
            receipt.parent / "decoded.json",
            "unified",
        )
        accept(images, checksum, pred)
    save_index(out, cfg, rows, images)
    return images


def save_index(out, cfg, rows, images):
    successful = sum(i["status"] == "ok" for i in images.values())
    atomic(
        out / "index.json",
        {
            "schema": "qwen-sgg-image-cache-v1",
            "protocol": protocol(cfg),
            "images": images,
            "unique_images": len(images),
            "successful_images": successful,
            "requested_pairs": len(rows),
            "execution_complete": successful == len(images),
        },
    )


def wrapped_record(pred, row, role):
    rec = copy.deepcopy(read(pred["record_path"]))
    identifier = row["task_id"] + "--" + role
    rec.update(candidate_id=identifier, image_id=identifier, pair_task_id=row["task_id"], image_role=role)
    rec["image"]["image_id"] = identifier
    rec["native_output"].update(
        raw_npz_path=pred["raw_path"], raw_npz_sha256=pred["raw_sha256"], decoded_path=pred["decoded_path"]
    )
    rec["provenance"]["shared_image_prediction"] = pred
    validate_record(rec)
    return rec


def pair_results(out, rows, images):
    pairs = []
    for row in rows:
        case = out / "pairs" / row["task_id"]
        case.mkdir(parents=True, exist_ok=True)
        pair = {k: row[k] for k in ("task_id", "image_id", "dataset", "prompt")}
        pair["roles"] = {}
        for role in ("source", "target"):
            checksum = image_hash(row, role)
            item = images[checksum]
            view = {
                "status": "not_run",
                "native_image_path": row[role + "_path"],
                "native_sha256": checksum,
                "prediction_key": item["key"],
                "graph_path": None,
            }
            if item["status"] == "ok":
                rec = wrapped_record(item["prediction"], row, role)
                path = case / (role + "-graph.json")
                if path.exists():
                    if read(path) != rec:
                        raise ValueError("refuse_to_overwrite_bound_pair_graph")
                else:
                    write(path, rec)
                view.update(
                    status="ok",
                    graph_path=str(path),
                    input=rec["image"],
                    graph_sha256=sha256(path),
                    prediction=item["prediction"],
                )
            pair["roles"][role] = view
        if (case / "receipt.json").exists():
            if bundle(case, "pair.json") != pair:
                raise ValueError("sealed_pair_changed")
        else:
            atomic(case / "pair.json", pair)
            if all(r["status"] == "ok" for r in pair["roles"].values()):
                finish(case)
        pairs.append(pair)
    atomic(out / "results.json", pairs)
    return pairs
