"""Real-image GQA GPU stress/throughput benchmark, with disposable model updates.

No quality scores or checkpoints are published. Existing pilot artifacts stay read-only.
"""

import argparse
import gc
import hashlib
import io
import json
import math
import os
import resource
import time
import zipfile
from pathlib import Path

import clip
import h5py
import numpy as np
import torch
from PIL import Image

from scripts.cache_vg150_regions import extract_vectors, scale_boxes_to_jpeg
from src.datasets.gqa import image_archive_index, normalize_gqa
from src.datasets.gqa_pilot import record_identity
from src.features.geometry import GeometryFeatureExtractor
from src.graph.builder import SceneGraphBuilder
from src.knowledge.transe.core import sha256
from src.training.gqa_pilot import collate, cross_entropy_sum, forward
from tools.prepare_level2_gqa import write_fixed
from tools.train_gqa_pilot import build_model, evaluate


def largest_ids(records, size):
    if not 0 < size <= len(records):
        raise ValueError("Invalid stress cohort size")
    return sorted(records, key=lambda i: (-len(records[i].objects), i))[:size]


def cached_graph(record, group):
    if group.attrs["record_sha256"] != record_identity(record):
        raise ValueError("Cached record/order mismatch")
    vectors = np.asarray(group["features"], dtype=np.float32)
    if vectors.shape != (len(record.objects), 512) or not np.isfinite(vectors).all():
        raise ValueError("Invalid CLIP vectors")
    x = torch.cat((GeometryFeatureExtractor().extract(record), torch.from_numpy(vectors)), dim=1)
    return SceneGraphBuilder().build(record, x)


def prepare(args):
    start = time.perf_counter()
    manifest = json.loads(args.manifest.read_text())
    source = args.root / "sceneGraphs/train_sceneGraphs.json"
    if sha256(source) != manifest["annotation_sha256"][source.name]:
        raise ValueError("Annotation changed")
    cohort = json.loads(args.pilot_cohort.read_text())
    if cohort["manifest_sha256"] != sha256(args.manifest):
        raise ValueError("Pilot manifest mismatch")
    if not set(cohort["splits"]["train"]) <= set(manifest["splits"]["train"]):
        raise ValueError("Pilot train outside project train")
    if not set(cohort["splits"]["val"]) <= set(manifest["splits"]["val"]):
        raise ValueError("Pilot val outside project val")
    raw = json.loads(source.read_text())
    # Select on normalized object count only, never relation labels or model scores.
    records = {i: normalize_gqa(i, raw[i], manifest["vocabulary"])[0] for i in manifest["splits"]["train"]}
    stress = largest_ids(records, max(args.batch_sizes))
    selected = {i: records[i] for i in set(stress) | set(cohort["splits"]["train"])}
    for i in cohort["splits"]["val"]:
        selected[i] = normalize_gqa(i, raw[i], manifest["vocabulary"])[0]
    del raw, records
    gc.collect()
    pin = {
        "manifest_sha256": sha256(args.manifest),
        "pilot_cohort_sha256": sha256(args.pilot_cohort),
        "selection": "largest normalized train object counts, descending; image_id ascending for ties",
        "stress_train_ids": stress,
        "typical_train_ids": cohort["splits"]["train"],
        "typical_val_ids": cohort["splits"]["val"],
        "test_annotation_opened": False,
    }
    write_fixed(args.report_dir / "cohort.json", pin)
    cache_identity = {
        "cohort_sha256": sha256(args.report_dir / "cohort.json"),
        "weights_sha256": sha256(args.weights),
        "encoder": "frozen OpenAI CLIP ViT-B/32; default transform; L2 normalized float16",
        "box_policy": "annotation xyxy scaled to JPEG, floor/ceil crop",
        "code_sha256": {
            s: sha256(Path(s))
            for s in (
                "tools/benchmark_gqa_scale.py",
                "scripts/cache_vg150_regions.py",
                "src/datasets/gqa.py",
                "src/datasets/gqa_pilot.py",
            )
        },
    }
    cache_manifest = args.cache / "manifest.json"
    feature_file = args.cache / "features.h5"
    args.cache.mkdir(parents=True, exist_ok=True)
    cache_start = time.perf_counter()
    if cache_manifest.exists():
        saved = json.loads(cache_manifest.read_text())
        if saved["identity"] != cache_identity or sha256(feature_file) != saved["features_sha256"]:
            raise ValueError("Stress cache identity/hash mismatch")
    else:
        if feature_file.exists():
            raise FileExistsError("Incomplete stress cache: use a fresh cache directory")
        model, transform = clip.load(str(args.weights), device="cuda", jit=False)
        model.eval().requires_grad_(False)
        archive = args.root / "archives/images.zip"
        names = image_archive_index(archive)
        with zipfile.ZipFile(archive) as z, h5py.File(feature_file, "x") as out:
            for image_id in stress:
                record = selected[image_id]
                jpeg = z.read(names[image_id])
                with Image.open(io.BytesIO(jpeg)) as im:
                    im = im.convert("RGB")
                    boxes = scale_boxes_to_jpeg(
                        [o.bbox_xyxy for o in record.objects], (record.width, record.height), im.size
                    )
                    vectors = extract_vectors(model, transform, im, boxes, 64, "cuda")
                group = out.create_group(image_id)
                group.create_dataset("features", data=vectors.astype(np.float16))
                group.attrs["record_sha256"] = record_identity(record)
                group.attrs["jpeg_sha256"] = hashlib.sha256(jpeg).hexdigest()
        torch.cuda.synchronize()
        write_fixed(
            cache_manifest,
            {
                "identity": cache_identity,
                "features_sha256": sha256(feature_file),
                "images": len(stress),
                "encoder_trainable_parameters": 0,
            },
        )
        del model, transform
        gc.collect()
        torch.cuda.empty_cache()
    cache_seconds = time.perf_counter() - cache_start
    pilot_metadata = json.loads((args.pilot_cache / "manifest.json").read_text())
    if pilot_metadata["cohort_sha256"] != sha256(args.pilot_cohort):
        raise ValueError("Pilot cache cohort mismatch")
    if sha256(args.pilot_cache / "features.h5") != pilot_metadata["features_sha256"]:
        raise ValueError("Pilot cache changed")
    loading_start = time.perf_counter()
    with h5py.File(feature_file) as stress_cache, h5py.File(args.pilot_cache / "features.h5") as pilot_cache:
        graphs = {
            "stress": [cached_graph(selected[i], stress_cache[i]) for i in stress],
            "typical_train": [cached_graph(selected[i], pilot_cache[i]) for i in cohort["splits"]["train"]],
            "typical_val": [cached_graph(selected[i], pilot_cache[i]) for i in cohort["splits"]["val"]],
        }
    return graphs, {
        "total_prepare_seconds": time.perf_counter() - start,
        "stress_cache_extract_or_verify_seconds": cache_seconds,
        "graph_build_and_hdf5_read_seconds": time.perf_counter() - loading_start,
        "host_ru_maxrss_bytes_linux_process_high_water": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "stress_cache_manifest_sha256": sha256(cache_manifest),
        "stress_graphs": [
            {
                "image_id": g.image_id,
                "objects": g.num_nodes,
                "candidate_pairs": g.edge_index.shape[1],
                "gt_rows": len(g.gt_predicate),
            }
            for g in graphs["stress"]
        ],
    }


def measurement(model, optimizer, kind, graphs, config, generator, training):
    torch.cuda.synchronize()
    start = time.perf_counter()
    batch, _ = collate(graphs, config["num_classes"], generator if training else None)
    collate_seconds = time.perf_counter() - start
    count = len(batch["targets"])
    if training:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = forward(model, kind, batch, "cuda")
        loss = cross_entropy_sum(logits, batch["targets"].cuda()) / max(count, 1)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
        finite = bool(torch.isfinite(loss)) and bool(torch.isfinite(norm))
    else:
        model.eval()
        with torch.inference_mode():
            logits = forward(model, kind, batch, "cuda")
            loss = cross_entropy_sum(logits, batch["targets"].cuda()) / max(count, 1)
        finite = bool(torch.isfinite(loss))
    torch.cuda.synchronize()
    if not finite:
        raise RuntimeError("Non-finite loss/gradients")
    return {
        "wall_seconds": time.perf_counter() - start,
        "collate_cpu_seconds": collate_seconds,
        "pairs_scored": count,
        "candidate_edges": sum(g.edge_index.shape[1] for g in graphs),
        "images": len(graphs),
        "loss_finite": finite,
    }


def run_case(kind, mode, graphs, batch_size, config, warmup, repeats):
    gc.collect()
    torch.cuda.empty_cache()
    torch.manual_seed(42)
    model = build_model(kind, config).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    generator = torch.Generator().manual_seed(42)
    training = mode != "stress_all_pairs_inference"
    groups = [graphs[i : i + batch_size] for i in range(0, len(graphs), batch_size)]
    result = {
        "model": kind,
        "mode": mode,
        "batch_size": batch_size,
        "images_per_repeat": len(graphs),
        "batches_per_repeat": len(groups),
        "warmup_batches": warmup,
        "repeats": repeats,
    }
    try:
        for _ in range(warmup):
            measurement(model, optimizer, kind, groups[0], config, generator, training)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        runs = []
        for _ in range(repeats):
            runs.append([measurement(model, optimizer, kind, group, config, generator, training) for group in groups])
        walls = [sum(row["wall_seconds"] for row in rows) for rows in runs]
        result.update(
            status="PASS",
            repeat_wall_seconds=walls,
            median_seconds=float(np.median(walls)),
            images_per_second=len(graphs) / float(np.median(walls)),
            collate_cpu_seconds_per_repeat=[sum(row["collate_cpu_seconds"] for row in rows) for rows in runs],
            pairs_scored_per_repeat=[sum(row["pairs_scored"] for row in rows) for rows in runs],
            candidate_edges_per_repeat=sum(g.edge_index.shape[1] for g in graphs),
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
            loss_and_gradient_finite=True,
        )
    except torch.cuda.OutOfMemoryError:
        result.update(
            status="OOM",
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    print(
        f"[BENCH] {kind} {mode} batch={batch_size}: {result['status']} "
        f"peak={result['peak_cuda_allocated_bytes']/2**30:.3f} GiB",
        flush=True,
    )
    return result


def validation_case(kind, graphs, batch_size, config, repeats):
    gc.collect()
    torch.cuda.empty_cache()
    torch.manual_seed(42)
    model = build_model(kind, config).cuda().eval()
    case_config = {**config, "batch_size": batch_size}
    # Includes CPU transfer, full per-image GC/NG ranking and aggregation. Scores are discarded.
    evaluate(model, kind, graphs[:batch_size], case_config, "cuda", include_ng=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = evaluate(model, kind, graphs, case_config, "cuda", include_ng=True)
        if not math.isfinite(result["loss_all_pairs"]):
            raise RuntimeError("Non-finite validation loss")
        torch.cuda.synchronize()
        elapsed.append(time.perf_counter() - start)
    row = {
        "model": kind,
        "mode": "typical_validation_gc_ng",
        "batch_size": batch_size,
        "status": "PASS",
        "images_per_repeat": len(graphs),
        "repeat_wall_seconds": elapsed,
        "median_seconds": float(np.median(elapsed)),
        "images_per_second": len(graphs) / float(np.median(elapsed)),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return row


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("data/raw/gqa"))
    p.add_argument("--manifest", type=Path, default=Path("data/processed/level2/gqa-v1/manifest.json"))
    p.add_argument("--pilot-cohort", type=Path, default=Path("reports/level2/gqa-pilot-v1-cohort.json"))
    p.add_argument("--pilot-cache", type=Path, default=Path("data/cache/gqa-pilot-clip-v1"))
    p.add_argument("--pilot-config", type=Path, default=Path("reports/level2/gqa-pilot-v1/config.json"))
    p.add_argument("--weights", type=Path, default=Path("data/cache/vg150-clip-vit-b32-v1/weights/ViT-B-32.pt"))
    p.add_argument("--cache", type=Path, default=Path("data/cache/gqa-benchmark-v1"))
    p.add_argument("--report-dir", type=Path, default=Path("reports/level2/gqa-benchmark-v1"))
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 16, 32, 64])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=2)
    args = p.parse_args(argv)
    if min(args.batch_sizes + [args.repeats, args.warmup]) <= 0:
        p.error("Positive batch sizes/repeats/warmup required")
    if (args.report_dir / "results.json").exists():
        raise FileExistsError("Use a new report directory for a new measurement")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    config = json.loads(args.pilot_config.read_text())
    for path, digest in config["code_sha256"].items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"Pilot dependency changed: {path}")
    args.report_dir.mkdir(parents=True, exist_ok=True)
    graphs, prep = prepare(args)
    cases = []
    for kind in ("mlp", "gat"):
        for size in sorted(set([1] + args.batch_sizes)):
            for mode in ("stress_sampled_train", "stress_all_pairs_inference"):
                cases.append(run_case(kind, mode, graphs["stress"][:size], size, config, args.warmup, args.repeats))
        for size in args.batch_sizes:
            cases.append(
                run_case(
                    kind, "typical_sampled_train", graphs["typical_train"], size, config, args.warmup, args.repeats
                )
            )
            cases.append(validation_case(kind, graphs["typical_val"], size, config, args.repeats))
    report = {
        "scope": "engineering benchmark only; disposable optimizer updates, no trained checkpoints or quality scores",
        "device": torch.cuda.get_device_name(),
        "total_device_bytes": torch.cuda.get_device_properties(0).total_memory,
        "torch_version": str(torch.__version__),
        "seed": 42,
        "dtype": "float32; no AMP or TF32",
        "deterministic_algorithms": True,
        "cpu_threads": torch.get_num_threads(),
        "warmup_batches": args.warmup,
        "repeats": args.repeats,
        "batch_sizes": args.batch_sizes,
        "timing": "perf_counter with CUDA synchronization before and after each measured batch; includes CPU collate and H2D",
        "model_config_sha256": sha256(args.pilot_config),
        "cohort_sha256": sha256(args.report_dir / "cohort.json"),
        "source_sha256": {
            p: sha256(Path(p))
            for p in ("tools/benchmark_gqa_scale.py", "tools/train_gqa_pilot.py", "src/training/gqa_pilot.py")
        },
        "prepare": prep,
        "cases": cases,
        "test_used": False,
        "checkpoint_saved": False,
        "memory_notes": "Peaks include model/AdamW state for training after warmup; allocated/reserved exclude CUDA context and other processes. Inference excludes optimizer state, which is lazy and uninitialized.",
    }
    write_fixed(args.report_dir / "results.json", report)
    print(f"[PASS] saved {len(cases)} cases to {args.report_dir}", flush=True)


if __name__ == "__main__":
    main()
