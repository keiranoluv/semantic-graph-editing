"""Bounded synthetic capacity/resume probe; no main training, no W&B."""

import argparse
import json
import time
from pathlib import Path

import torch

from experiments.aligng_adapted_v1.batching import collate, model_inputs
from experiments.aligng_adapted_v1.training import batch_loss
from src.datasets.gqa_full import atomic_write, pin_json
from src.datasets.schemas import SceneGraphRecord, SceneObject, SceneRelation
from src.graph.builder import SceneGraphBuilder
from tools.run_gqa_full import exclusive

from . import engine
from .runner import ROOT, source_identity


def synthetic(nodes, seed=42):
    record = SceneGraphRecord(
        str(seed),
        512,
        512,
        tuple(SceneObject(i, i % 1699 + 1, (i, i, i + 10, i + 10)) for i in range(nodes)),
        (SceneRelation(0, 1, 310), SceneRelation(0, 1, 1)),
    )
    return SceneGraphBuilder().build(record, torch.randn(nodes, 517, generator=torch.Generator().manual_seed(seed)))


def main(argv=None):
    from dataclasses import asdict

    from experiments.aligng_adapted_v1 import AlignGConfig

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    identity = source_identity()
    dimensions = {"semantic_dim": 256, "pe_dropout": 0.2, "aligng": asdict(AlignGConfig())}
    with exclusive(ROOT / "runs/level2/.gqa-kg-training-device"):
        engine.setup_runtime({"device": args.device, "num_threads": 1})
        pin_json(args.output / "identity.json", {**identity, "device": args.device, "seed": 42})
        results = []
        for batch_size in (1, 8, 16):
            torch.manual_seed(42)
            model = engine.create_model(
                "pe-plpr-aligng", dimensions, [torch.randn(1700, 300), torch.randn(311, 300)]
            ).to(args.device)
            batch = collate([synthetic(126, 42 + i) for i in range(batch_size)], 311, torch.Generator().manual_seed(42))
            if args.device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            start = time.perf_counter()
            loss = batch_loss(model, batch, args.device)["total"]
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            if args.device == "cuda":
                torch.cuda.synchronize()
            results.append(
                {
                    "batch_size": batch_size,
                    "objects_per_image": 126,
                    "context_pairs_per_image": 15750,
                    "loss": float(loss.detach()),
                    "grad_norm": float(norm),
                    "seconds": time.perf_counter() - start,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
                }
            )
            assert torch.isfinite(loss) and all(
                p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
            )
            del loss, model, batch
        # Real model, deterministic CUDA resume with dropout; tiny synthetic training only.
        from .engine import train_run
        from .runner import DEFAULT, load_config

        config = load_config(DEFAULT)
        config.update(device=args.device, epochs=2, train_batch_size=2, val_batch_size=1, checkpoint_every_batches=1)
        study = {"config": config, "dimensions": {**dimensions, "num_classes": 311}, **identity}
        graphs = [synthetic(3, seed) for seed in (42, 43, 44)]
        vectors = [torch.randn(1700, 300), torch.randn(311, 300)]
        full = train_run("pe-plpr-aligng", 42, graphs, graphs[:1], vectors, study, args.output / "continuous")
        if not (args.output / "resumed/last.pt").exists():
            assert (
                train_run(
                    "pe-plpr-aligng",
                    42,
                    graphs,
                    graphs[:1],
                    vectors,
                    study,
                    args.output / "resumed",
                    stop_after_batches=1,
                )
                is None
            )
        resumed = train_run("pe-plpr-aligng", 42, graphs, graphs[:1], vectors, study, args.output / "resumed")
        for key in ("history", "validation", "selected_epoch"):
            assert full[key] == resumed[key], key
        a, b = [torch.load(args.output / name / "last.pt", weights_only=True) for name in ("continuous", "resumed")]
        for key in a["model"]:
            torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
        # GPU isolation + full-query largest image forward.
        model = engine.create_model("pe-plpr-aligng", dimensions, vectors).to(args.device).eval()
        with torch.no_grad():
            alone = model(**model_inputs(collate(graphs[:1], 311), args.device))
            together = model(**model_inputs(collate(graphs[:2], 311), args.device))[: len(alone)]
            torch.testing.assert_close(alone, together, atol=2e-6, rtol=2e-6)
            out = model(**model_inputs(collate([synthetic(126)], 311), args.device))
            assert out.shape == (15750, 311) and torch.isfinite(out).all()
        result = {
            "status": "PASS",
            "device": args.device,
            "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
            "capacity": results,
            "exact_resume": True,
            "isolation": True,
            "full_query_15750_finite": True,
            "test_annotations_used": False,
            "main_training_started": False,
            **identity,
        }
        atomic_write(args.output / "result.json", result)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
