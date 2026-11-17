"""Bounded capacity checks only: disposable optimizer steps, no main training/W&B."""

import gc
import json
import time
from pathlib import Path

import torch

from experiments.aligng_fast_v1 import batching as fast_batching
from src.datasets.gqa_full import GQAFullDataset, atomic_write, digest
from src.knowledge.transe.core import sha256
from src.training.gqa_pilot import cross_entropy_sum
from src.training.penet import prototype_losses
from tools.run_gqa_full import exclusive

from . import engine
from .runner import CASES, MODELS, PROBE, ROOT, load_config, prepare


def measure(model_name, graphs, training, exp, vectors, device="cuda", warmup=1, repeats=2, *, variant="fast"):
    if variant != "fast":
        raise ValueError("Unknown execution variant")
    module = engine
    batching = fast_batching
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
                loss = prototype_losses(
                    logits,
                    relation,
                    prototypes,
                    batch["targets"].to(device),
                    mode=engine.VARIANTS[model_name]["loss_mode"],
                )["total"]
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
        raise FileExistsError("Probe evidence already exists; do not overwrite")
    cfg = load_config()
    exp, metadata, vectors = prepare(cfg)
    pinned = json.loads(CASES.read_text())
    if pinned["cache_manifest_sha256"] != sha256(Path(cfg["cache"]) / "manifest.json"):
        raise ValueError("Capacity case cache changed")
    rows = []
    with exclusive(ROOT / "runs/level2/.gqa-kg-training-device"):
        engine.setup_runtime(cfg)
        for name, case in pinned["cases"].items():
            dataset = GQAFullDataset(cfg["cache"], metadata, case["split"])
            try:
                graphs = [dataset[i] for i in case["indices"]]
            finally:
                dataset.close()
            if len(graphs) != 512:
                raise ValueError("Capacity case must contain 512 images")
            for model in MODELS:
                print(f"[PROBE] {model} {name} batch512", flush=True)
                row = measure(model, graphs, case["split"] == "train", exp, vectors)
                row.pop("_logits")
                row.update(model=model, case=name, images=512)
                rows.append(row)
                print(json.dumps(row), flush=True)
    atomic_write(
        PROBE,
        {
            "status": "PASS",
            "experiment_sha256": digest(exp),
            "case_source_sha256": sha256(CASES),
            "results": rows,
            "gpu": torch.cuda.get_device_name(),
            "seed": 42,
            "warmup": 1,
            "measured_steps": 2,
            "test_used": False,
            "wandb_used": False,
            "training_checkpoint_saved": False,
            "scope": "disposable capacity optimizer steps only; six sequential model/case pairs",
        },
    )


if __name__ == "__main__":
    main()
