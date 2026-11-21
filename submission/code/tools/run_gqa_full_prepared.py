"""Resume GQA with exact graph and fixed supervision preprocessing cached in RAM."""

import argparse
import io
import json
import os
from pathlib import Path

import torch

from src.datasets.gqa_full import GQAFullDataset, atomic_write
from src.knowledge.transe.core import sha256
from src.training.gqa_pilot import soft_targets
from tools import run_gqa_full_ram as ram


def prepare_supervision(graph, num_classes):
    # Compute using the frozen reference, retaining float32 values bit for bit.
    target = soft_targets(graph, num_classes)
    positive = torch.where(target[:, 0] == 0)[0]
    negative = torch.where(target[:, 0] == 1)[0]
    graph.prepared_positive = positive
    graph.prepared_negative = negative
    graph.prepared_rows = target[positive].clone()
    return graph


def prepared_collate(graphs, num_classes, generator=None):
    """Same candidate order, randperm calls and targets as the frozen collator."""
    xs, labels, edges, pairs, targets, slices = [], [], [], [], [], []
    node_offset, pair_offset = 0, 0
    for graph in graphs:
        positive, negative, rows = graph.prepared_positive, graph.prepared_negative, graph.prepared_rows
        if rows.shape != (len(positive), num_classes):
            raise ValueError("Prepared supervision vocabulary mismatch")
        if generator is not None:
            count = min(len(negative), 3 * len(positive) if len(positive) else 32)
            chosen = negative[torch.randperm(len(negative), generator=generator)[:count]]
            selected = torch.cat((positive, chosen))
            target = torch.zeros((len(selected), num_classes), dtype=torch.float32)
            target[: len(positive)] = rows
            target[len(positive) :, 0] = 1
        else:
            selected = torch.arange(graph.edge_label_index.shape[1], dtype=torch.long)
            target = torch.zeros((len(selected), num_classes), dtype=torch.float32)
            target[:, 0] = 1
            target[positive] = rows
        xs.append(graph.x)
        labels.append(graph.object_labels)
        edges.append(graph.edge_index + node_offset)
        pairs.append(graph.edge_label_index[:, selected] + node_offset)
        targets.append(target)
        slices.append((pair_offset, pair_offset + len(selected)))
        node_offset += graph.num_nodes
        pair_offset += len(selected)
    if not graphs:
        raise ValueError("Cannot collate an empty batch")
    return {
        "x": torch.cat(xs),
        "object_labels": torch.cat(labels),
        "edge_index": torch.cat(edges, dim=1),
        "pairs": torch.cat(pairs, dim=1),
        "targets": torch.cat(targets),
    }, slices


def load_prepared_shard(directory, entry, split):
    torch.set_num_threads(1)
    metadata = {"identity": {"splits": {split: entry["ids"]}}, "shards": [entry]}
    dataset = GQAFullDataset(directory, metadata, split)
    try:
        items = [prepare_supervision(dataset[i], 311).to_dict() for i in range(len(dataset))]
    finally:
        dataset.close()
    stream = io.BytesIO()
    torch.save(items, stream)
    return stream.getvalue(), ram.graph_digest(items)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--loader-workers", type=int, default=8)
    parser.add_argument("--config", type=Path, default=Path("configs/level2/gqa-full-v3-bs512.json"))
    args, remaining = parser.parse_known_args(argv)
    if not 1 <= args.loader_workers <= (os.cpu_count() or 1):
        parser.error("loader-workers must be between 1 and the logical CPU count")
    from src.training import gqa_full
    from src.training.gqa_tracking import RunTracker, preserve_rng
    from tools import run_gqa_full as launcher
    from tools import train_gqa_pilot as evaluator

    config = launcher.load_config(args.config)
    sources = [Path(__file__), Path(ram.__file__)]
    hashes = {str(p.relative_to(Path.cwd())): sha256(p) for p in sources}
    runtime = {
        "scope": "performance-only prepared supervision; exact sampling/targets; frozen training recipe unchanged",
        "pid": os.getpid(),
        "workers": args.loader_workers,
        "source_sha256": hashes,
        "supervision": "original soft_targets float32 positive rows and candidate index lists; backgrounds implicit",
        "datasets": {},
    }

    def dataset_factory(directory, metadata, split):
        output = Path(config["output"] + ("-smoke" if metadata["identity"]["scope"] == "smoke" else ""))
        report = output / f"runtime-prepared-loader-{os.getpid()}.json"
        atomic_write(report, runtime)
        dataset = ram.RAMGQADataset(directory, metadata, split, workers=args.loader_workers)
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
                            "runtime/loader": "CPU workers preload graphs and fixed supervision into RAM",
                            "runtime/loader_workers": args.loader_workers,
                            "runtime/prepared_source_sha256": hashes,
                        }
                    )

    original = (launcher.GQAFullDataset, gqa_full.RunTracker, gqa_full.collate, evaluator.collate, ram.load_shard)
    launcher.GQAFullDataset, gqa_full.RunTracker = dataset_factory, RuntimeTracker
    gqa_full.collate = evaluator.collate = prepared_collate
    ram.load_shard = load_prepared_shard
    try:
        print(f"[PREPARED] {json.dumps(runtime)}", flush=True)
        return launcher.main(["--config", str(args.config), *remaining])
    finally:
        launcher.GQAFullDataset, gqa_full.RunTracker, gqa_full.collate, evaluator.collate, ram.load_shard = original


if __name__ == "__main__":
    raise SystemExit(main())
