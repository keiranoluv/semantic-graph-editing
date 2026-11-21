"""Check all five models on demanding actual 512-image batches before a fresh run.

Select max candidate/sample counts from first-epoch shuffles of the three seeds,
and max candidate count from fixed validation order. Not an all-epochs bound.
Updates are disposable and no training checkpoint or W&B run is written.
"""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.datasets.gqa import normalize_gqa
from src.datasets.gqa_full import GQAFullDataset, atomic_write, verify_cache
from src.knowledge.transe.core import sha256
from src.training.gqa_full import create_model, load_vectors, model_config, setup_runtime
from src.training.gqa_pilot import collate, cross_entropy_sum, forward
from src.training.penet import prototype_losses
from tools.run_gqa_full import load_config


def select_cases(config, manifest):
    source = Path(config["root"]) / "sceneGraphs/train_sceneGraphs.json"
    if sha256(source) != manifest["annotation_sha256"][source.name]:
        raise ValueError("Annotation mismatch")
    raw = json.loads(source.read_text())
    counts = {}
    for split in ("train", "val"):
        rows = []
        for image_id in manifest["splits"][split]:
            record = normalize_gqa(image_id, raw[image_id], manifest["vocabulary"])[0]
            n = len(record.objects)
            pairs = n * (n - 1)
            positive = len({(r.subject_id, r.object_id) for r in record.relations})
            sampled = positive + min(pairs - positive, 3 * positive if positive else 32)
            rows.append((pairs, sampled))
        counts[split] = np.array(rows)
    choices = {}
    for seed in config["seeds"]:
        order = torch.randperm(len(counts["train"]), generator=torch.Generator().manual_seed(seed)).numpy()
        for begin in range(0, len(order), 512):
            indices = order[begin : begin + 512]
            total = counts["train"][indices].sum(0)
            for column, name in enumerate(("train_max_candidates", "train_max_sampled")):
                if name not in choices or total[column] > choices[name]["selection_count"]:
                    choices[name] = {
                        "split": "train",
                        "seed": seed,
                        "batch_index": begin // 512,
                        "indices": indices.tolist(),
                        "selection_count": int(total[column]),
                    }
    for begin in range(0, len(counts["val"]), 512):
        total = int(counts["val"][begin : begin + 512, 0].sum())
        if "val_max_candidates" not in choices or total > choices["val_max_candidates"]["selection_count"]:
            choices["val_max_candidates"] = {
                "split": "val",
                "batch_index": begin // 512,
                "indices": list(range(begin, min(begin + 512, len(counts["val"])))),
                "selection_count": total,
            }
    return choices


def measure(kind, graphs, training, config, dimensions, vectors):
    gc.collect()
    torch.cuda.empty_cache()
    torch.manual_seed(42)
    model = create_model(kind, dimensions, vectors).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    generator = torch.Generator().manual_seed(42)
    model.train(training)

    def step():
        torch.cuda.synchronize()
        start = time.perf_counter()
        batch, _ = collate(graphs, dimensions["num_classes"], generator if training else None)
        optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            if training and kind.startswith("pe-"):
                logits, rel, proto = model(
                    batch["x"].cuda(), batch["pairs"].cuda(), batch["object_labels"].cuda(), return_embeddings=True
                )
                loss = prototype_losses(logits, rel, proto, batch["targets"].cuda(), mode=kind[3:])["total"]
            else:
                logits = forward(model, "penet" if kind.startswith("pe-") else kind, batch, "cuda")
                loss = cross_entropy_sum(logits, batch["targets"].cuda()) / len(logits)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss")
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config["gradient_clip_norm"], error_if_nonfinite=True
                )
                optimizer.step()
        torch.cuda.synchronize()
        return {"wall_seconds": time.perf_counter() - start, "pairs_scored": len(logits)}

    try:
        for _ in range(2):
            step()
        torch.cuda.reset_peak_memory_stats()
        rows = [step() for _ in range(3)]
        return {
            "status": "PASS",
            "measurements": rows,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
    except torch.cuda.OutOfMemoryError:
        return {"status": "OOM"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/level2/gqa-full-v3-bs512.json"))
    p.add_argument("--output", type=Path, default=Path("reports/level2/gqa-batch512-v1/benchmark.json"))
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("Use fresh output")
    config = load_config(args.config)
    if config["train_batch_size"] != 512 or config["val_batch_size"] != 512 or config["device"] != "cuda":
        raise ValueError("Expected CUDA batch 512/512")
    setup_runtime(config)
    manifest = json.loads(Path(config["manifest"]).read_text())
    metadata = verify_cache(config["cache"])
    if metadata["identity"]["manifest_sha256"] != sha256(Path(config["manifest"])):
        raise ValueError("Cache vocabulary mismatch")
    cases = select_cases(config, manifest)
    dimensions = model_config(manifest)
    vectors = load_vectors(config, dimensions)
    results = []
    for name, case in cases.items():
        dataset = GQAFullDataset(config["cache"], metadata, case["split"])
        try:
            start = time.perf_counter()
            graphs = [dataset[i] for i in case["indices"]]
            load_seconds = time.perf_counter() - start
        finally:
            dataset.close()
        for kind in config["models"]:
            result = measure(kind, graphs, case["split"] == "train", config, dimensions, vectors)
            result.update(
                model=kind,
                case=name,
                images=len(graphs),
                objects=sum(g.num_nodes for g in graphs),
                candidate_pairs=sum(g.edge_index.shape[1] for g in graphs),
                graph_load_seconds=load_seconds,
            )
            results.append(result)
            print(json.dumps(result), flush=True)
    atomic_write(
        args.output,
        {
            "scope": __doc__,
            "cases": cases,
            "results": results,
            "gpu": torch.cuda.get_device_name(),
            "warmup_batches": 2,
            "measured_batches": 3,
            "config_sha256": sha256(args.config),
            "cache_manifest_sha256": sha256(Path(config["cache"]) / "manifest.json"),
            "source_sha256": {"tools/benchmark_gqa_batch512.py": sha256(Path(__file__))},
            "test_used": False,
            "checkpoint_saved": False,
            "wandb_used": False,
        },
    )
    if any(r["status"] != "PASS" for r in results):
        raise RuntimeError("At least one batch failed; do not start the matrix")


if __name__ == "__main__":
    main()
