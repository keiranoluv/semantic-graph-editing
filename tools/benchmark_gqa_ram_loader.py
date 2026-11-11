"""CPU-only real-cache parity/throughput probe; safe beside the running trainer."""

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import torch

from src.datasets.gqa_full import GQAFullDataset, atomic_write, verify_cache
from src.knowledge.transe.core import sha256
from src.training.gqa_pilot import collate
from tools.run_gqa_full_ram import RAMGQADataset, graph_digest


def measure(dataset):
    generator = torch.Generator().manual_seed(42)
    h = hashlib.sha256()
    seconds, pairs = 0.0, 0
    for begin in range(0, len(dataset), 512):
        start = time.perf_counter()
        batch, _ = collate(dataset[begin : begin + 512], 311, generator)
        seconds += time.perf_counter() - start
        # Verification is outside the timed region for both paths.
        for key in sorted(batch):
            h.update(key.encode())
            h.update(batch[key].numpy().tobytes())
        pairs += len(batch["targets"])
    return {"loader_and_collate_seconds": seconds, "sampled_pairs": pairs, "batch_tensor_sha256": h.hexdigest()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--images", type=int, default=4096)
    p.add_argument("--output", type=Path, default=Path("reports/level2/gqa-ram-loader-v1/benchmark.json"))
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("Use fresh output")
    torch.set_num_threads(1)
    directory = Path("data/cache/gqa-full-clip-v1")
    metadata = verify_cache(directory)
    ids = metadata["identity"]["splits"]["train"]
    if not 0 < args.images <= len(ids):
        p.error("Invalid image count")
    order = torch.randperm(len(ids), generator=torch.Generator().manual_seed(42)).tolist()[: args.images]
    selected = copy.deepcopy(metadata)
    selected["identity"]["splits"]["train"] = [ids[i] for i in order]
    lazy = GQAFullDataset(directory, selected, "train")
    ram = RAMGQADataset(directory, selected, "train", workers=args.workers)
    try:
        for i in range(len(ram)):
            if graph_digest([ram[i].to_dict()]) != graph_digest([lazy[i].to_dict()]):
                raise AssertionError(f"Graph differs at {i}")
        results = {"lazy": [], "ram": []}
        for _ in range(3):
            for name, dataset in (("lazy", lazy), ("ram", ram)):
                row = measure(dataset)
                results[name].append(row)
                print(name, json.dumps(row), flush=True)
        if len({r["batch_tensor_sha256"] for rows in results.values() for r in rows}) != 1:
            raise AssertionError("Prepared batch tensors differ")
        means = {k: sum(r["loader_and_collate_seconds"] for r in v) / len(v) for k, v in results.items()}
        report = {
            "status": "PASS",
            "scope": "CPU loader+collate only; not end-to-end training speedup",
            "images": args.images,
            "workers": args.workers,
            "batch_size": 512,
            "selection": "first images of seed42 first-epoch shuffle, no label/metric filtering",
            "all_graph_tensors_equal": True,
            "all_sampled_batch_tensors_equal": True,
            "ram_preload": ram.statistics,
            "measurements": results,
            "mean_seconds": means,
            "loader_collate_speedup": means["lazy"] / means["ram"],
            "cache_manifest_sha256": sha256(directory / "manifest.json"),
            "source_sha256": {
                str(Path(s)): sha256(Path(s))
                for s in ("tools/benchmark_gqa_ram_loader.py", "tools/run_gqa_full_ram.py")
            },
            "gpu_used": False,
            "test_used": False,
        }
        atomic_write(args.output, report)
        print(json.dumps({k: report[k] for k in ("status", "mean_seconds", "loader_collate_speedup")}), flush=True)
    finally:
        lazy.close()
        ram.close()


if __name__ == "__main__":
    main()
