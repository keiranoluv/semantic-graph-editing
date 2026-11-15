"""Prepare/preflight/baseline/smoke/infer/quality: isolated pretrained five-image SGG."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from tools.level3_image_graph_v1.core import (
    DEFAULT_LOCKED,
    ROOT,
    digest,
    load_pilot,
    outside_locked,
    read,
    sha256,
    validate_record,
    write,
)
from tools.level3_image_graph_v1.runner import read_bundle

from . import VERSION
from .adapter import ParseError, robin_record
from .assets import CACHE, prepare_assets, runtime_versions, verify_assets
from .backend import RobinBackend, load_sam, propose_regions
from .comparison import export_comparison, validate_quality
from .cuda_check import cuda_device_evidence, kernel_probe

CONFIG = ROOT / "configs/level3/sgg-compare-v1.json"
BASELINE = ROOT / "runs/level3/image-graph-v1/pilot30"


def config_at(path):
    config = read(path)
    if (
        config["schema"] != VERSION
        or config["batch_size"] != 1
        or config["training"] is not False
        or config["automatic_queue"] is not False
        or len(config["images"]) != 5
        or len({r["candidate_id"] for r in config["images"]}) != 5
    ):
        raise ValueError("invalid_five_image_protocol")
    if not 1 <= config["sam"]["max_regions"] <= 99:
        raise ValueError("invalid_region_budget")
    if not 1 <= config["robin"]["max_new_tokens"] <= 4096:
        raise ValueError("invalid_decode_budget")
    if (
        not 1 <= config["threads"] <= 64
        or config["sam"]["model_type"] != "vit_h"
        or not 0 <= config["sam"]["min_area_fraction"] < config["sam"]["max_area_fraction"] <= 1
        or config["robin"]["dtype"] != "bfloat16"
    ):
        raise ValueError("invalid_runtime_or_region_policy")
    if (
        config["robin"]["coordinate_policy"] != "aligned_square_pad_v1"
        or config["robin"]["attribute_policy"] != "leading_explicit_modifiers_v1"
        or config["robin"]["do_sample"] is not False
        or config["robin"]["num_beams"] != 1
    ):
        raise ValueError("unsupported_inference_policy")
    return config


def inputs(args):
    cfg = config_at(args.config)
    manifest, identity = load_pilot(args.locked)
    if (
        identity["manifest_sha256"] != cfg["locked_manifest_sha256"]
        or identity["tasks_sha256"] != cfg["locked_tasks_sha256"]
    ):
        raise ValueError("locked_cohort_revision_mismatch")
    for file, key in (("candidates.json", "baseline_candidates_sha256"), ("receipt.json", "baseline_receipt_sha256")):
        if sha256(args.baseline / file) != cfg[key]:
            raise ValueError("baseline_revision_mismatch")
    source = {r["image_id"]: r for r in manifest["images"]}
    bundle = read_bundle(args.baseline)
    if bundle["run"]["mode"] != "infer" or len(bundle["records"]) != 30:
        raise ValueError("baseline_must_be_completed_real_pilot30")
    old = {r["candidate_id"]: r for r in bundle["records"]}
    rows, baseline = [], []
    for selected in cfg["images"]:
        row = source[selected["candidate_id"]]
        if row["sha256"] != selected["sha256"] or old[row["image_id"]]["image"] != row:
            raise ValueError("selected_image_or_baseline_mismatch")
        rows.append(row)
        record = copy.deepcopy(old[row["image_id"]])
        validate_record(record)
        record.update(
            model="reltr-clip",
            status="ok",
            comparison_origin={
                "baseline_candidates_sha256": cfg["baseline_candidates_sha256"],
                "original_record_sha256": digest(old[row["image_id"]]),
            },
        )
        baseline.append(record)
    return cfg, rows, baseline, identity


def protected_output(path, args):
    for root in (DEFAULT_LOCKED, args.locked, args.baseline, args.cache):
        outside_locked(path, root)
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError("immutable_output_exists; use a new versioned output")
    return path


def code_identity():
    paths = list(Path(__file__).parent.glob("*.py")) + [
        ROOT / "src/editing/operations.py",
        ROOT / "tools/level3_image_graph_v1/core.py",
        ROOT / "tools/level3_image_graph_v1/runner.py",
    ]
    return {
        "commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "files_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)},
    }


def finish(output, mode, counts):
    write(
        output / "receipt.json",
        {
            "schema": VERSION,
            "mode": mode,
            "execution_complete": True,
            "status_counts": counts,
            "human_quality_measured": False,
            "files_sha256": {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()},
        },
    )


def baseline_only(args):
    cfg, rows, baseline, cohort = inputs(args)
    output = protected_output(args.output, args)
    output.mkdir(parents=True)
    write(
        output / "cohort.json", {"config_sha256": sha256(args.config), "parent": cohort, "images": rows, "config": cfg}
    )
    counts = export_comparison(rows, baseline, [], args.locked, output)
    finish(output, "baseline_only_robin_not_run", counts)
    print(json.dumps({"output": str(output), "images": len(rows), "status": counts}))


def set_seed(cfg, device):
    import torch

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(cfg["threads"])
    if device == "cuda":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("ROBIN requires available BF16 CUDA for this launcher; no fallback")
        torch.cuda.manual_seed_all(cfg["seed"])
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def release_cuda(device):
    import torch

    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


def infer(args):
    output = protected_output(args.output, args)
    device_evidence = cuda_device_evidence() if args.device == "cuda" else {"device": "cpu"}
    cfg, rows, baseline, cohort = inputs(args)
    assets = verify_assets(args.cache)
    versions = runtime_versions()
    set_seed(cfg, args.device)
    output.mkdir(parents=True)
    run = {
        "schema": VERSION,
        "mode": "infer",
        "device": args.device,
        "device_evidence": device_evidence,
        "config": cfg,
        "config_sha256": sha256(args.config),
        "assets": assets,
        "runtime": versions,
        "code": code_identity(),
        "cohort": cohort,
        "images": rows,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "training": False,
        "automatic_queue": False,
        "sequence": ["SAM all five images", "unload SAM", "ROBIN all five images"],
        "external_image_uploads": False,
        "baseline_reinferred": False,
    }
    write(output / "run.json", run)
    sampler = load_sam(args.cache / "sam_vit_h_4b8939.pth", cfg["sam"], args.device)
    proposals = {}
    for row in rows:
        with Image.open(args.locked / row["image_path"]) as image:
            proposal = propose_regions(sampler, image.convert("RGB"), cfg["sam"])
        proposal.update(candidate_id=row["image_id"], source_sha256=row["sha256"])
        write(output / "regions" / (row["image_id"] + ".json"), proposal)
        proposals[row["image_id"]] = proposal
        print(
            json.dumps({"stage": "sam", "candidate_id": row["image_id"], "regions": len(proposal["regions"])}),
            flush=True,
        )
    del sampler
    release_cuda(args.device)
    robin = RobinBackend(args.cache / "SyntheticVG", args.cache / "robin", args.device)
    write(output / "model-loading.json", robin.loading_evidence)
    records = []
    for row in rows:
        identifier = row["image_id"]
        regions = proposals[identifier]["regions"]
        provenance = {"run_sha256": digest(run), "regions_sha256": sha256(output / "regions" / (identifier + ".json"))}
        if not regions:
            records.append(
                {
                    "candidate_id": identifier,
                    "image": row,
                    "source_sha256": row["sha256"],
                    "status": "no_regions",
                    "source_graph": None,
                    "provenance": provenance,
                }
            )
            continue
        with Image.open(args.locked / row["image_path"]) as image:
            text, generation = robin.generate(image.convert("RGB"), regions, cfg["robin"])
        native = {"candidate_id": identifier, "source_sha256": row["sha256"], "text": text, "generation": generation}
        raw_path = output / "native" / (identifier + ".json")
        write(raw_path, native)
        provenance["native_sha256"] = sha256(raw_path)
        try:
            record = robin_record(row, regions, text, provenance, generation)
            validate_record(record)
            record["status"] = "ok"
        except ParseError as error:
            record = {
                "candidate_id": identifier,
                "image": row,
                "source_sha256": row["sha256"],
                "status": "parse_error",
                "source_graph": None,
                "error": str(error),
                "native_output": native,
                "provenance": provenance,
            }
        records.append(record)
        print(json.dumps({"stage": "robin", "candidate_id": identifier, "status": record["status"]}), flush=True)
    del robin
    release_cuda(args.device)
    if inputs(args)[3] != cohort:
        raise ValueError("inputs_changed_during_inference")
    write(output / "robin.json", {"schema": VERSION, "records": records})
    counts = export_comparison(rows, baseline, records, args.locked, output)
    finish(output, "infer", counts)
    print(json.dumps({"output": str(output), "status": counts, "quality": "pending_human_review"}))
    return 0 if all(r["status"] == "ok" for r in records) else 2


def smoke(args):
    """Explicit bounded runtime check, NOT five-image inference or graph evaluation."""
    import torch
    from pycocotools import mask as mask_utils

    output = protected_output(args.output, args)
    device = "cuda" if args.command == "gpu-smoke" else "cpu"
    device_evidence = cuda_device_evidence() if device == "cuda" else {"device": "cpu"}
    cfg, rows, _, _ = inputs(args)
    assets = verify_assets(args.cache)
    runtime = runtime_versions()
    set_seed(cfg, device)
    output.mkdir(parents=True)
    probe = kernel_probe() if device == "cuda" else None
    row = rows[0]
    sam_config = {**cfg["sam"], "points_per_side": 1, "points_per_batch": 1, "crop_n_layers": 0}
    sam = load_sam(args.cache / "sam_vit_h_4b8939.pth", sam_config, device)
    sam_frozen = all(not p.requires_grad for p in sam.predictor.model.parameters())
    if device == "cuda":
        with Image.open(args.locked / row["image_path"]) as image:
            proposal = propose_regions(sam, image.convert("RGB"), sam_config)
        write(output / "sam-probe.json", proposal)
    del sam
    release_cuda(device)
    robin = RobinBackend(args.cache / "SyntheticVG", args.cache / "robin", device)
    w, h = row["width"], row["height"]
    rle = mask_utils.encode(np.asfortranarray(np.ones((h, w), dtype=np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    region = {"region_id": 1, "bbox_xyxy": [0, 0, w, h], "segmentation": rle}
    with Image.open(args.locked / row["image_path"]) as image:
        text, evidence = robin.generate(image.convert("RGB"), [region], cfg["robin"], max_new_tokens=8)
    write(
        output / "smoke.json",
        {
            "mode": f"{device}_runtime_smoke_not_quality_evaluation",
            "device": device,
            "device_evidence": device_evidence,
            "kernel_probe": probe,
            "sam_forward_executed": device == "cuda",
            "sam_smoke_config": sam_config,
            "candidate_id": row["image_id"],
            "source_sha256": row["sha256"],
            "assets": assets,
            "runtime": runtime,
            "code": code_identity(),
            "config_sha256": sha256(args.config),
            "sam_weights_strictly_loaded_and_frozen": sam_frozen,
            "robin": robin.loading_evidence,
            "region_origin": "whole_image_fixture_not_sam_prediction",
            "text": text,
            "generation": evidence,
            "inference_mode": True,
            "torch_cuda_initialized": torch.cuda.is_initialized(),
            "graph_quality_measured": False,
        },
    )
    finish(output, f"{device}_runtime_smoke_not_quality_evaluation", {})
    print(
        json.dumps(
            {
                "output": str(output),
                "device": device,
                "new_tokens": evidence["generated_tokens"],
                "quality_measured": False,
            }
        )
    )


def verify_receipt(path):
    path = Path(path).resolve()
    receipt = read(path / "receipt.json")
    if receipt.get("execution_complete") is not True:
        raise ValueError("incomplete_comparison_bundle")
    for name, checksum in receipt["files_sha256"].items():
        file = (path / name).resolve()
        if path not in file.parents or sha256(file) != checksum:
            raise ValueError("comparison_bundle_changed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "preflight", "baseline", "infer", "smoke", "gpu-smoke", "quality"):
        p = sub.add_parser(name)
        p.add_argument("--config", type=Path, default=CONFIG)
        p.add_argument("--locked", type=Path, default=DEFAULT_LOCKED)
        p.add_argument("--baseline", type=Path, default=BASELINE)
        p.add_argument("--cache", type=Path, default=CACHE)
        if name in {"baseline", "infer", "smoke", "gpu-smoke", "quality"}:
            p.add_argument("--output", type=Path, required=True)
        if name == "infer":
            p.add_argument("--device", choices=("cpu", "cuda"), required=True)
        if name == "quality":
            p.add_argument("--bundle", type=Path, required=True)
            p.add_argument("--review", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        for root in (DEFAULT_LOCKED, args.locked, args.baseline):
            outside_locked(args.cache, root)
        print(json.dumps(prepare_assets(args.cache)))
    elif args.command == "preflight":
        cfg, rows, _, cohort = inputs(args)
        evidence = verify_assets(args.cache)
        print(
            json.dumps(
                {
                    "passed": True,
                    "candidate_ids": [r["image_id"] for r in rows],
                    "cohort": cohort,
                    "assets_lock_sha256": evidence["lock_sha256"],
                    "runtime": runtime_versions(),
                    "config_sha256": digest(cfg),
                    "gpu_used": False,
                }
            )
        )
    elif args.command == "baseline":
        baseline_only(args)
    elif args.command == "infer":
        return infer(args)
    elif args.command in {"smoke", "gpu-smoke"}:
        smoke(args)
    else:
        output = protected_output(args.output, args)
        outside_locked(output, args.bundle)
        verify_receipt(args.bundle)
        summary = validate_quality(read(args.bundle / "comparison.json"), read(args.review))
        write(output, {**summary, "submitted_review": read(args.review)})
        print(json.dumps(summary))
    return 0
