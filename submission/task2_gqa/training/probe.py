"""Bounded actual-data batch512 capacity check; no main checkpoint or W&B."""

import gc
import json
import time
from pathlib import Path

import torch

from task2_gqa.models import batching as old_batching
from task2_gqa.recipe import engine as old_engine
from common.datasets.gqa_full import GQAFullDataset, atomic_write, digest
from common.knowledge.transe.core import sha256
from common.training.gqa_pilot import cross_entropy_sum
from common.training.penet import prototype_losses
from task2_gqa.run_gqa_full import exclusive

from task2_gqa.training import batching as fast_batching
from task2_gqa.training import engine
from task2_gqa.training.runner import MODELS, PROBE, ROOT, load_config, prepare

# Audited max-context, max-sampled and max-validation batches, selected using
# first-epoch shuffles of seeds42/123/3407. No held-out test, no metric tuning.
CASES = ROOT / "reports/level2/gqa-batch512-v1/benchmark.json"


def measure(model_name, graphs, training, exp, vectors, device="cuda", warmup=1, repeats=2, *, variant="fast"):
    if variant not in ("fast", "original"):
        raise ValueError("Unknown execution variant")
    module = engine if variant == "fast" else old_engine
    batching = fast_batching if variant == "fast" else old_batching
    collate, model_inputs = batching.collate, batching.model_inputs
    if variant == "fast":
        graphs = [
            g
            if isinstance(g, fast_batching.PreparedGraph)
            else fast_batching.prepare_graph(g, exp["dimensions"]["num_classes"])
            for g in graphs
        ]
    cfg = exp["config"]
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    torch.manual_seed(42)
    model = module.create_model(model_name, exp["dimensions"], vectors).to(device)
    model.train(training)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    generator = torch.Generator().manual_seed(42)
    captured = {}

    def step():
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        batch = collate(graphs, exp["dimensions"]["num_classes"], generator if training else None)
        collate_seconds = time.perf_counter() - start
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
            seconds = time.perf_counter() - start
            if not training:
                captured["logits"] = logits.detach().cpu()
            return {
                "seconds": seconds,
                "collate_cpu_seconds": collate_seconds,
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
        "variant": variant,
        "_logits": captured.get("logits"),
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
                comparisons = {}
                reference = None
                for variant in ("original", "fast"):
                    print(f"[PROBE] {name} {model} {variant} 512 images", flush=True)
                    result = measure(model, graphs, case["split"] == "train", exp, vectors, variant=variant)
                    logits = result.pop("_logits")
                    if variant == "original":
                        reference = logits
                    elif logits is not None:
                        torch.testing.assert_close(logits, reference, rtol=1e-4, atol=2e-4)
                        result["eval_equivalence"] = {
                            "max_abs_logit_difference": float((logits - reference).abs().max()),
                            "rtol": 1e-4,
                            "atol": 2e-4,
                            "status": "PASS",
                        }
                    comparisons[variant] = result
                    print(json.dumps({"variant": variant, **result}), flush=True)
                result = comparisons["fast"]
                result.update(
                    model=model,
                    case=name,
                    images=len(graphs),
                    objects=sum(g.num_nodes for g in graphs),
                    context_pairs=sum(g.edge_index.shape[1] for g in graphs),
                    original=comparisons["original"],
                    speedup=sum(r["seconds"] for r in comparisons["original"]["measurements"])
                    / sum(r["seconds"] for r in result["measurements"]),
                )
                results.append(result)
                del reference, logits
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
            "scope": "disposable capacity updates only; sequential original/fast, same initialization and samples; validation forward/CE excludes CPU ranking; one-time CPU graph preparation excluded",
        },
    )


if __name__ == "__main__":
    main()
