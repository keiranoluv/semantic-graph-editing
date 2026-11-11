"""CPU-only benchmark of exact prepared supervision against the RAM-only loader."""

import copy
import hashlib
import json
import time
from pathlib import Path

import torch

from src.datasets.gqa_full import atomic_write, verify_cache
from src.knowledge.transe.core import sha256
from src.training.gqa_pilot import collate
from tools import run_gqa_full_ram as ram
from tools.run_gqa_full_prepared import load_prepared_shard, prepared_collate


def measure(dataset, function, training):
    generator = torch.Generator().manual_seed(42) if training else None
    h = hashlib.sha256()
    seconds = 0.0
    for begin in range(0, len(dataset), 512):
        start = time.perf_counter()
        batch, slices = function(dataset[begin : begin + 512], 311, generator)
        seconds += time.perf_counter() - start
        for key in sorted(batch):
            h.update(key.encode())
            h.update(batch[key].numpy().tobytes())
        h.update(json.dumps(slices).encode())
    if generator is not None:
        h.update(generator.get_state().numpy().tobytes())
    return {"seconds": seconds, "batches_and_sampler_sha256": h.hexdigest()}


def main():
    output = Path("reports/level2/gqa-prepared-loader-v1/benchmark.json")
    if output.exists():
        raise FileExistsError("Benchmark evidence already exists")
    torch.set_num_threads(1)
    metadata = verify_cache("data/cache/gqa-full-clip-v1")
    selected = copy.deepcopy(metadata)
    ids = metadata["identity"]["splits"]["train"]
    indices = torch.randperm(len(ids), generator=torch.Generator().manual_seed(42)).tolist()[:4096]
    selected["identity"]["splits"]["train"] = [ids[i] for i in indices]
    original = ram.load_shard
    ram.load_shard = load_prepared_shard
    try:
        dataset = ram.RAMGQADataset("data/cache/gqa-full-clip-v1", selected, "train", workers=8)
    finally:
        ram.load_shard = original
    measurements = []
    try:
        for training in (True, False):
            for threads in (1, 2, 4, 8):
                torch.set_num_threads(threads)
                for repeat in range(3):
                    for name, function in (("reference", collate), ("prepared", prepared_collate)):
                        row = {
                            "training": training,
                            "threads": threads,
                            "repeat": repeat,
                            "method": name,
                            **measure(dataset, function, training),
                        }
                        measurements.append(row)
                        print(json.dumps(row), flush=True)
            rows = [r for r in measurements if r["training"] == training]
            assert len({r["batches_and_sampler_sha256"] for r in rows}) == 1
        atomic_write(
            output,
            {
                "status": "PASS",
                "scope": "CPU collation only, not end-to-end training speedup",
                "images": 4096,
                "batch_size": 512,
                "preload": dataset.statistics,
                "all_batch_tensors_slices_and_sampler_states_equal": True,
                "measurements": measurements,
                "source_sha256": {
                    str(p): sha256(p)
                    for p in map(
                        Path,
                        [
                            "tools/benchmark_gqa_prepared.py",
                            "tools/run_gqa_full_prepared.py",
                            "tools/run_gqa_full_ram.py",
                        ],
                    )
                },
                "test_used": False,
                "gpu_used": False,
            },
        )
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
