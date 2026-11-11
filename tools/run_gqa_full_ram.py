"""Performance-only launcher: preload identical GQA graphs into RAM with CPU workers.

The frozen trainer/config/source stay unchanged so existing checkpoints can resume.
This explicit runtime overlay is recorded separately, including its source hash.
No shuffle, sampling, optimizer or numerical training implementation is replaced.
"""

import argparse
import hashlib
import io
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
from torch_geometric.data import Data

from src.datasets.gqa_full import GQAFullDataset, atomic_write
from src.knowledge.transe.core import sha256


def graph_digest(items):
    h = hashlib.sha256()
    for item in items:
        for key, value in sorted(item.items()):
            h.update(key.encode())
            if isinstance(value, torch.Tensor):
                h.update(str((str(value.dtype), tuple(value.shape))).encode())
                h.update(value.contiguous().numpy().tobytes())
            else:
                h.update(json.dumps(value, sort_keys=True).encode())
    return h.hexdigest()


def load_shard(directory, entry, split):
    # Spawn, not fork: the parent may have already initialized CUDA in preflight.
    # One Torch thread per worker avoids multiplying thread pools by worker count.
    torch.set_num_threads(1)
    metadata = {"identity": {"splits": {split: entry["ids"]}}, "shards": [entry]}
    dataset = GQAFullDataset(directory, metadata, split)
    try:
        items = [dataset[i].to_dict() for i in range(len(dataset))]
    finally:
        dataset.close()
    stream = io.BytesIO()
    torch.save(items, stream)
    # Bytes IPC avoids shared-memory/file-descriptor exhaustion from many tiny tensors.
    return stream.getvalue(), graph_digest(items)


class RAMGQADataset:
    def __init__(self, directory, metadata, split, workers=8):
        if split not in ("train", "val") or workers < 1:
            raise ValueError("Expected train/val and positive worker count")
        start = time.perf_counter()
        self.ids = metadata["identity"]["splits"][split]
        needed = set(self.ids)
        entries = [
            {**entry, "ids": [i for i in entry["ids"] if i in needed]}
            for entry in metadata["shards"]
            if set(entry["ids"]) & needed
        ]
        self.graphs = [None] * len(self.ids)
        positions = {image_id: i for i, image_id in enumerate(self.ids)}
        payload_bytes = 0
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            # Bound in-flight serialized shards to workers, not the whole dataset.
            pending = {}
            next_submit = 0
            for done in range(len(entries)):
                while next_submit < min(done + workers, len(entries)):
                    entry = entries[next_submit]
                    pending[next_submit] = pool.submit(load_shard, str(directory), entry, split)
                    next_submit += 1
                payload, expected = pending.pop(done).result()
                items = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
                if graph_digest(items) != expected:
                    raise ValueError("Graph tensor serialization changed values/order")
                payload_bytes += len(payload)
                for item in items:
                    image_id = item["image_id"]
                    if image_id in positions:
                        index = positions[image_id]
                        if self.graphs[index] is not None:
                            raise ValueError("Duplicate graph in cache shards")
                        self.graphs[index] = Data.from_dict(item)
                if (done + 1) % 16 == 0 or done + 1 == len(entries):
                    print(f"[RAM] {split}: {done + 1}/{len(entries)} shards loaded", flush=True)
        if any(graph is None for graph in self.graphs):
            raise ValueError("Missing preloaded graph")
        self.statistics = {
            "split": split,
            "images": len(self.ids),
            "workers": workers,
            "serialized_shard_bytes": payload_bytes,
            "tensor_bytes": sum(
                t.numel() * t.element_size()
                for g in self.graphs
                for t in g.to_dict().values()
                if isinstance(t, torch.Tensor)
            ),
            "load_seconds": time.perf_counter() - start,
            "all_worker_parent_tensor_digests_equal": True,
        }
        print(f"[RAM READY] {json.dumps(self.statistics)}", flush=True)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        return self.graphs[index]

    def close(self):
        self.graphs.clear()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--loader-workers", type=int, default=8)
    parser.add_argument("--config", type=Path, default=Path("configs/level2/gqa-full-v3-bs512.json"))
    args, remaining = parser.parse_known_args(argv)
    if not 1 <= args.loader_workers <= (os.cpu_count() or 1):
        parser.error("loader-workers must be between 1 and the logical CPU count")
    # Import the existing launcher only in the parent; worker imports do no GPU work.
    from src.training import gqa_full
    from src.training.gqa_tracking import RunTracker, preserve_rng
    from tools import run_gqa_full as launcher

    config = launcher.load_config(args.config)
    source = Path(__file__)
    runtime = {
        "scope": "performance-only RAM graph loading; training recipe and checkpoint identity unchanged",
        "pid": os.getpid(),
        "workers": args.loader_workers,
        "source_sha256": {str(source.relative_to(Path.cwd())): sha256(source)},
        "dataset_implementation": "original GQAFullDataset in spawned CPU workers; all tensor bytes verified after IPC",
        "datasets": {},
    }

    def dataset_factory(directory, metadata, split):
        output = Path(config["output"] + ("-smoke" if metadata["identity"]["scope"] == "smoke" else ""))
        report = output / f"runtime-ram-loader-{os.getpid()}.json"
        atomic_write(report, runtime)
        dataset = RAMGQADataset(directory, metadata, split, workers=args.loader_workers)
        runtime["datasets"][split] = dataset.statistics
        atomic_write(report, runtime)
        return dataset

    class RuntimeTracker(RunTracker):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            if self.run is not None:
                with preserve_rng():
                    self.run.summary.update(
                        {
                            "runtime/loader": "CPU worker preload into RAM",
                            "runtime/loader_workers": args.loader_workers,
                            "runtime/loader_source_sha256": sha256(source),
                        }
                    )

    original_dataset, original_tracker = launcher.GQAFullDataset, gqa_full.RunTracker
    launcher.GQAFullDataset, gqa_full.RunTracker = dataset_factory, RuntimeTracker
    try:
        return launcher.main(["--config", str(args.config), *remaining])
    finally:
        launcher.GQAFullDataset, gqa_full.RunTracker = original_dataset, original_tracker


if __name__ == "__main__":
    raise SystemExit(main())
