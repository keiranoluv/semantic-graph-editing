"""Measure PE-PLPR on the pinned largest GQA train graphs; updates are disposable."""

import argparse
import gc
import json
import os
import time
from pathlib import Path

import h5py
import torch

from src.datasets.gqa import normalize_gqa
from src.knowledge.transe.core import sha256
from src.models.penet import PENetPredicateClassifier
from src.training.gqa_pilot import collate
from src.training.penet import prototype_losses
from tools.benchmark_gqa_scale import cached_graph
from tools.prepare_level2_gqa import write_fixed


def measure(model, optimizer, graphs, config, generator, training):
    torch.cuda.synchronize()
    start = time.perf_counter()
    batch, _ = collate(graphs, config["num_classes"], generator if training else None)
    model.train(training)
    optimizer.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(training):
        logits, relations, prototypes = model(
            batch["x"].cuda(), batch["pairs"].cuda(), batch["object_labels"].cuda(), return_embeddings=True
        )
        if not torch.isfinite(logits).all():
            raise RuntimeError("Non-finite logits")
        if training:
            losses = prototype_losses(logits, relations, prototypes, batch["targets"].cuda(), mode="plpr")
            if not torch.isfinite(losses["total"]):
                raise RuntimeError("Non-finite loss")
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
    torch.cuda.synchronize()
    return {"wall_seconds": time.perf_counter() - start, "pairs_scored": len(logits)}


def run_case(saved, graphs, training):
    gc.collect()
    torch.cuda.empty_cache()
    torch.manual_seed(42)
    config, state = saved["config"], saved["model"]
    model = PENetPredicateClassifier(
        state["object_embedding.weight"],
        state["predicate_embedding.weight"],
        semantic_dim=config["semantic_dim"],
        dropout=config["dropout"],
    ).cuda()
    model.load_state_dict(state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    generator = torch.Generator().manual_seed(42)
    result = {
        "batch_size": len(graphs),
        "mode": "sampled_train_plpr" if training else "all_pairs_forward",
        "objects": sum(g.num_nodes for g in graphs),
        "candidate_pairs": sum(g.edge_index.shape[1] for g in graphs),
        "warmup_batches": 2,
        "measured_batches": 3,
    }
    try:
        for _ in range(2):
            measure(model, optimizer, graphs, config, generator, training)
        torch.cuda.reset_peak_memory_stats()
        rows = [measure(model, optimizer, graphs, config, generator, training) for _ in range(3)]
        result.update(
            status="PASS",
            measurements=rows,
            mean_wall_seconds=sum(r["wall_seconds"] for r in rows) / len(rows),
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
    except torch.cuda.OutOfMemoryError:
        result.update(status="OOM")
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot-report", type=Path, default=Path("reports/level2/penet-pilot-v1/plpr.json"))
    p.add_argument("--manifest", type=Path, default=Path("data/processed/level2/gqa-v1/manifest.json"))
    p.add_argument("--annotation", type=Path, default=Path("data/raw/gqa/sceneGraphs/train_sceneGraphs.json"))
    p.add_argument("--cohort", type=Path, default=Path("reports/level2/gqa-benchmark-v1/cohort.json"))
    p.add_argument("--cache", type=Path, default=Path("data/cache/gqa-benchmark-v1"))
    p.add_argument("--output", type=Path, default=Path("reports/level2/penet-pilot-v1/stress.json"))
    args = p.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Use a fresh benchmark output")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)
    report, manifest, cohort, cache = [
        json.loads(f.read_text()) for f in (args.pilot_report, args.manifest, args.cohort, args.cache / "manifest.json")
    ]
    if (
        sha256(args.manifest) != report["config"]["manifest_sha256"]
        or sha256(args.manifest) != cohort["manifest_sha256"]
    ):
        raise ValueError("Manifest mismatch")
    if sha256(args.annotation) != manifest["annotation_sha256"][args.annotation.name]:
        raise ValueError("Annotation mismatch")
    if cache["identity"]["cohort_sha256"] != sha256(args.cohort):
        raise ValueError("Stress cohort mismatch")
    if sha256(args.cache / "features.h5") != cache["features_sha256"]:
        raise ValueError("Stress cache mismatch")
    for name, expected in cache["identity"]["code_sha256"].items():
        if sha256(Path(name)) != expected:
            raise ValueError(f"Cache source changed: {name}")
    ids = cohort["stress_train_ids"]
    if len(ids) != 64 or not set(ids) <= set(manifest["splits"]["train"]):
        raise ValueError("Expected 64 pinned project-train images")
    checkpoint = Path(report["checkpoint"])
    if sha256(checkpoint) != report["checkpoint_sha256"]:
        raise ValueError("Checkpoint mismatch")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["config"] != report["config"] or saved["variant"] != "plpr":
        raise ValueError("Unexpected checkpoint config/variant")
    raw = json.loads(args.annotation.read_text())
    with h5py.File(args.cache / "features.h5") as features:
        graphs = [cached_graph(normalize_gqa(i, raw[i], manifest["vocabulary"])[0], features[i]) for i in ids]
    del raw
    gc.collect()
    cases = []
    for training in (True, False):
        for size in (1, 8, 32, 64):
            row = run_case(saved, graphs[:size], training)
            cases.append(row)
            print(json.dumps(row), flush=True)
    if sha256(checkpoint) != report["checkpoint_sha256"]:
        raise AssertionError("Checkpoint changed")
    write_fixed(
        args.output,
        {
            "scope": "PE-PLPR stress memory/latency; disposable updates; no quality metrics or saved training",
            "gpu": torch.cuda.get_device_name(),
            "torch_version": str(torch.__version__),
            "pilot_report_sha256": sha256(args.pilot_report),
            "cohort_sha256": sha256(args.cohort),
            "cache_manifest_sha256": sha256(args.cache / "manifest.json"),
            "code_sha256": {str(Path(__file__).relative_to(Path.cwd())): sha256(Path(__file__))},
            "cases": cases,
            "checkpoint_unchanged": True,
            "test_used": False,
        },
    )


if __name__ == "__main__":
    main()
