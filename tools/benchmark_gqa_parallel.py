"""Disposable two-process GPU throughput/parity probe on actual GQA batches."""

import hashlib
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import torch

from src.datasets.gqa_full import GQAFullDataset, atomic_write
from src.training.gqa_full import create_model, load_vectors, model_config, setup_runtime
from src.training.gqa_pilot import cross_entropy_sum, forward
from src.training.penet import prototype_losses
from tools.run_gqa_full import load_config
from tools.run_gqa_full_prepared import prepare_supervision, prepared_collate
from tools.train_penet_pilot import state_hash


def worker(index, cpus, incoming, outgoing):
    os.sched_setaffinity(0, cpus)
    config = load_config("configs/level2/gqa-full-v3-bs512.json")
    setup_runtime(config)
    dimensions = model_config(json.loads(Path(config["manifest"]).read_text()))
    vectors = load_vectors(config, dimensions)
    metadata = json.loads(Path(config["cache"], "manifest.json").read_text())
    ds = GQAFullDataset(config["cache"], metadata, "train")
    indices = torch.randperm(len(ds), generator=torch.Generator().manual_seed(42)).tolist()[:4096]
    graphs = [prepare_supervision(ds[i], 311) for i in indices]
    ds.close()
    outgoing.put((index, "ready"))
    while (task := incoming.get()) is not None:
        try:
            kind, seed = task
            torch.manual_seed(seed)
            generator = torch.Generator().manual_seed(seed)
            model = create_model(kind, dimensions, vectors).cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
            model.train()
            trace = ""
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            for step in range(24):
                group = graphs[(step % 8) * 512 : (step % 8 + 1) * 512]
                batch, _ = prepared_collate(group, 311, generator)
                h = hashlib.sha256(trace.encode())
                h.update(json.dumps([g.image_id for g in group]).encode())
                h.update(batch["pairs"].numpy().tobytes())
                h.update(batch["targets"].numpy().tobytes())
                trace = h.hexdigest()
                optimizer.zero_grad(set_to_none=True)
                if kind.startswith("pe-"):
                    logits, relations, prototypes = model(
                        batch["x"].cuda(), batch["pairs"].cuda(), batch["object_labels"].cuda(), return_embeddings=True
                    )
                    loss = prototype_losses(logits, relations, prototypes, batch["targets"].cuda(), mode=kind[3:])[
                        "total"
                    ]
                else:
                    logits = forward(model, kind, batch, "cuda")
                    loss = cross_entropy_sum(logits, batch["targets"].cuda()) / len(logits)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                optimizer.step()
                float(loss.detach()), float(norm)
            torch.cuda.synchronize()
            result = {
                "kind": kind,
                "seed": seed,
                "seconds": time.perf_counter() - start,
                "model_sha256": state_hash(model),
                "trace": trace,
                "sampler_sha256": hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest(),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            }
            del model, optimizer, loss, logits, batch
            if kind.startswith("pe-"):
                del relations, prototypes
            torch.cuda.empty_cache()
            outgoing.put((index, result))
        except Exception as error:
            outgoing.put((index, {"error": repr(error)}))
            raise


def main():
    output = Path("reports/level2/gqa-parallel-v1/benchmark.json")
    if output.exists():
        raise FileExistsError(output)
    ctx = mp.get_context("spawn")
    cpus = sorted(os.sched_getaffinity(0))[:16]
    queues, results = [ctx.Queue() for _ in range(2)], ctx.Queue()
    processes = [ctx.Process(target=worker, args=(i, cpus[i * 8 : (i + 1) * 8], queues[i], results)) for i in range(2)]
    rows = []
    try:
        for process in processes:
            process.start()
        for _ in processes:
            print(results.get(timeout=120), flush=True)
        for kinds in (("mlp", "mlp"), ("gat", "pe-plpr"), ("pe-plpr", "pe-plpr")):
            case = {"models": kinds}
            for concurrent in (False, True):
                start = time.perf_counter()
                values = {}
                for i, kind in enumerate(kinds):
                    queues[i].put((kind, 42 + i))
                    if not concurrent:
                        slot, value = results.get(timeout=180)
                        values[slot] = value
                if concurrent:
                    for _ in processes:
                        slot, value = results.get(timeout=180)
                        values[slot] = value
                assert all("error" not in v for v in values.values()), values
                case["parallel" if concurrent else "sequential"] = {
                    "wall_seconds": time.perf_counter() - start,
                    "runs": values,
                }
                print(json.dumps(case), flush=True)
            for i in range(2):
                for key in ("model_sha256", "trace", "sampler_sha256"):
                    assert case["parallel"]["runs"][i][key] == case["sequential"]["runs"][i][key]
            case["speedup"] = case["sequential"]["wall_seconds"] / case["parallel"]["wall_seconds"]
            rows.append(case)
        atomic_write(
            output,
            {
                "status": "PASS",
                "scope": "24 training batches per task; no validation; existing production run active in background",
                "cases": rows,
                "exact_model_sampling_parity": True,
                "cpu_affinity": cpus,
                "test_used": False,
            },
        )
    finally:
        for queue in queues:
            queue.put(None)
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join()


if __name__ == "__main__":
    main()
