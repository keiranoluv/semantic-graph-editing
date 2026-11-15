"""Bounded actual-data batch512 capacity check; no main checkpoint or W&B."""

import gc
import json
import time
from pathlib import Path

import torch

from experiments.aligng_adapted_v1.batching import collate, model_inputs
from experiments.aligng_gqa_v1 import engine
from src.datasets.gqa_full import GQAFullDataset, atomic_write, digest
from src.knowledge.transe.core import sha256
from src.training.gqa_pilot import cross_entropy_sum
from src.training.penet import prototype_losses
from tools.run_gqa_full import exclusive

from .runner import MODELS, PROBE, ROOT, load_config, prepare

# Audited max-context, max-sampled and max-validation batches, selected using
# first-epoch shuffles of seeds42/123/3407. No held-out test, no metric tuning.
CASES = ROOT / "reports/level2/gqa-batch512-v1/benchmark.json"


def measure(model_name, graphs, training, exp, vectors, device="cuda", warmup=1, repeats=2):
    cfg = exp["config"]
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    torch.manual_seed(42)
    model = engine.create_model(model_name, exp["dimensions"], vectors).to(device)
    model.train(training)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    generator = torch.Generator().manual_seed(42)

    def step():
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        batch = collate(graphs, exp["dimensions"]["num_classes"], generator if training else None)
        optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits, relation, prototypes = model(**model_inputs(batch, device), return_embeddings=True)
            if training:
                loss = prototype_losses(logits, relation, prototypes, batch["targets"].to(device))["total"]
            else:
                loss = cross_entropy_sum(logits, batch["targets"].to(device)) / len(logits)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite batch512 loss")
            norm = None
            if training:
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg["gradient_clip_norm"], error_if_nonfinite=True
                )
                optimizer.step()
            if device == "cuda":
                torch.cuda.synchronize()
            return {
                "seconds": time.perf_counter() - start,
                "loss": float(loss.detach()),
                "gradient_norm": float(norm) if norm is not None else None,
                "query_pairs": len(logits),
            }

    for _ in range(warmup):
        step()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    rows = [step() for _ in range(repeats)]
    return {
        "status": "PASS",
        "measurements": rows,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved() if device == "cuda" else None,
    }


def main():
    if PROBE.exists():
        raise FileExistsError("Capacity evidence is immutable; use a new version for changed code")
    cfg = load_config()
    exp, metadata, vectors = prepare(cfg)
    pinned = json.loads(CASES.read_text())
    if pinned["cache_manifest_sha256"] != sha256(Path(cfg["cache"]) / "manifest.json"):
        raise ValueError("Historical capacity cases refer to a different cache")
    results = []
    with exclusive(ROOT / "runs/level2/.gqa-kg-training-device"):
        engine.setup_runtime(cfg)
        for name, case in pinned["cases"].items():
            dataset = GQAFullDataset(cfg["cache"], metadata, case["split"])
            try:
                graphs = [dataset[i] for i in case["indices"]]
            finally:
                dataset.close()
            if len(graphs) != 512:
                raise ValueError("Expected exactly 512 real images")
            for model in MODELS:
                print(f"[PROBE] {name} {model} 512 images", flush=True)
                result = measure(model, graphs, case["split"] == "train", exp, vectors)
                result.update(
                    model=model,
                    case=name,
                    images=len(graphs),
                    objects=sum(g.num_nodes for g in graphs),
                    context_pairs=sum(g.edge_index.shape[1] for g in graphs),
                )
                results.append(result)
                print(json.dumps(result), flush=True)
    atomic_write(
        PROBE,
        {
            "status": "PASS",
            "experiment_sha256": digest(exp),
            "results": results,
            "gpu": torch.cuda.get_device_name(),
            "cases": pinned["cases"],
            "case_source_sha256": sha256(CASES),
            "warmup": 1,
            "measured_steps": 2,
            "seed": 42,
            "test_used": False,
            "wandb_used": False,
            "training_checkpoint_saved": False,
            "scope": "disposable capacity updates only, not full training; validation forward/CE, excludes CPU ranking",
        },
    )


if __name__ == "__main__":
    main()
