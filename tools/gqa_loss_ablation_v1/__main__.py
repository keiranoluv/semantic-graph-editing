import argparse
import json
import signal
import statistics
import tempfile
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch

from src.datasets.gqa_full import atomic_write, digest, pin_json, verify_cache
from src.knowledge.transe.core import sha256
from src.training import gqa_full as trainer
from src.training.gqa_pilot import cross_entropy_sum
from tools import run_gqa_full as historical
from tools import run_gqa_full_prepared as prepared
from tools import run_gqa_full_ram as ram
from tools import train_gqa_pilot as evaluator

CONFIG = Path("configs/level2/gqa-full-v3-bs512.json")
OUT = Path("runs/level2/gqa-gat-loss-v1")
TAU = 0.25


class LossTracker(trainer.RunTracker):
    def __init__(self, experiment, identity, directory):
        super().__init__(experiment, identity, directory)
        if self.run is not None:
            self.run.name = f"gqa-gat-{experiment['condition']}-seed{identity['seed']}"


def training_prior(train, classes):
    mass = torch.zeros(classes, dtype=torch.float64)
    for graph in train:
        positive = graph.prepared_rows
        mass += positive.sum(dim=0, dtype=torch.float64)
        budget = 3 * len(positive) if len(positive) else 32
        mass[0] += min(len(graph.prepared_negative), budget)
    smoothed = mass + 1
    return {
        "split": "train",
        "images": len(train),
        "target_mass": mass.tolist(),
        "add_one": 1,
        "tau": TAU,
        "prior": (smoothed / smoothed.sum()).tolist(),
    }


def paired_summary(results):
    deltas = []
    for seed, (control, adjusted) in results.items():
        if control["initial_state_sha256"] != adjusted["initial_state_sha256"]:
            raise ValueError("Unmatched initial weights")
        for a, b in zip(control["history"], adjusted["history"], strict=False):
            if a["sampling_trace_sha256"] != b["sampling_trace_sha256"]:
                raise ValueError("Unmatched training pairs or targets")
        delta = {
            k: 100 * (adjusted["validation"]["gc"][k] - control["validation"]["gc"][k])
            for k in ("R@50", "mR@50", "R@100", "mR@100")
        }
        deltas.append({"seed": seed, "delta_pp": delta})
    summary = {"split": "validation", "per_seed": deltas}
    if len(deltas) > 1:
        summary["paired_delta_pp"] = {
            k: {
                "mean": statistics.mean(r["delta_pp"][k] for r in deltas),
                "sample_sd": statistics.stdev(r["delta_pp"][k] for r in deltas),
            }
            for k in deltas[0]["delta_pp"]
        }
    return summary


def datasets(config, metadata, smoke, stack):
    if smoke:
        size = 512 if config["device"] == "cuda" else 16
        splits = {k: metadata["identity"]["splits"][k][:size] for k in ("train", "val")}
        metadata = {**metadata, "identity": {**metadata["identity"], "splits": splits}}
    groups = []
    with patch.object(ram, "load_shard", prepared.load_prepared_shard):
        for split in ("train", "val"):
            ds = ram.RAMGQADataset(config["cache"], metadata, split, workers=4)
            stack.callback(ds.close)
            groups.append(ds)
    return groups


def run(output, device, smoke, stop):
    if sha256(CONFIG) != "7a1e9b3d768df9dcf8abc72028431da6305e8ac088e6bbdb6426acf0a200fe22":
        raise ValueError("Expected GQA v3 training recipe")
    config = historical.load_config(CONFIG)
    cache_path = Path(config["cache"])
    for path, expected in [
        (cache_path / "manifest.json", "e337266e93fc36822ad13d0023f5cdddf9a80b9a9f32818343298710b3fc6407"),
        (Path(config["manifest"]), "c953dc40d4b0a880ccd142f36670388ebffb7d7e2bb341d661b8b45781046c2a"),
    ]:
        if sha256(path) != expected:
            raise ValueError(f"Frozen GQA input changed: {path}")
    config.update(output=str(output), device=device, models=["gat"])
    if smoke:
        config.update(epochs=1, seeds=[42], scope="smoke")
        config["wandb"]["mode"] = "disabled"
    trainer.setup_runtime(config)
    manifest = json.loads(Path(config["manifest"]).read_text())
    metadata = verify_cache(cache_path)
    for split in ("train", "val"):
        if metadata["identity"]["splits"][split] != manifest["splits"][split]:
            raise ValueError("Cache and dataset split differ")
    experiment = historical.experiment_identity(config, manifest)
    experiment["protocol"] = "GQA PredCls GAT KG-off; raw logits for validation GC/NG; fixed sampled soft targets"
    experiment["code_sha256"].update(
        {
            str(p): sha256(p)
            for p in [
                Path(__file__).resolve().relative_to(Path.cwd()),
                Path("scripts/run_gqa_loss_ablation_v1.sh"),
                Path(prepared.__file__).resolve().relative_to(Path.cwd()),
                Path(ram.__file__).resolve().relative_to(Path.cwd()),
            ]
        }
    )
    with ExitStack() as stack:
        train, val = datasets(config, metadata, smoke, stack)
        prior = training_prior(train, 311)
        pin_json(output / "train-prior.json", prior)
        shift = TAU * torch.tensor(prior["prior"], dtype=torch.float32, device=device).log()
        stack.enter_context(patch.object(trainer, "collate", prepared.prepared_collate))
        stack.enter_context(patch.object(evaluator, "collate", prepared.prepared_collate))
        stack.enter_context(patch.object(trainer, "RunTracker", LossTracker))
        result_pairs = {}
        print(
            f"[READY] train={len(train)} val={len(val)} device={device} batch=512 "
            f"epochs={config['epochs']} seeds={config['seeds']} tau={TAU}",
            flush=True,
        )
        for seed in config["seeds"]:
            pair = []
            for condition in ("ce", "adjusted"):
                if stop():
                    return
                directory = output / condition
                spec = {
                    **experiment,
                    "condition": condition,
                    "prior_sha256": digest(prior),
                    "config": {**config, "output": str(directory)},
                }
                pin_json(directory / "experiment.json", spec)
                # Only the training loop references trainer.cross_entropy_sum.
                # evaluator.evaluate imports its own CE and always sees raw logits.
                loss = cross_entropy_sum if condition == "ce" else lambda z, y: cross_entropy_sum(z + shift, y)
                print(f"[TRAIN] GAT/{condition} seed={seed}", flush=True)
                with patch.object(trainer, "cross_entropy_sum", loss):
                    result = trainer.train_run(
                        "gat", seed, train, val, None, spec, directory / str(seed), stop_requested=stop
                    )
                if result is None:
                    return
                pair.append(result)
                print(
                    f"[DONE] {condition} seed={seed} best_epoch={result['selected_epoch']} "
                    f"peak_allocated_GiB={(result['peak_cuda_allocated_bytes_this_invocation'] or 0) / 2**30:.2f}",
                    flush=True,
                )
            result_pairs[seed] = pair
            summary = paired_summary(result_pairs)
            atomic_write(output / "validation-comparison.json", summary)
            print("[PAIRED]", summary["per_seed"][-1], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "smoke-cpu", "smoke-gpu"))
    stage = parser.parse_args().stage
    if stage != "train":
        with tempfile.TemporaryDirectory(prefix="gqa-gat-loss-smoke-") as temp:
            run(Path(temp), "cpu" if stage == "smoke-cpu" else "cuda", True, lambda: False)
        print("[CLEAN] Temporary smoke directory and checkpoints removed.", flush=True)
        return
    halted = False

    def stop_requested(*_):
        nonlocal halted
        halted = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop_requested)
    with historical.exclusive(OUT):
        run(OUT, "cuda", False, lambda: halted)
    if halted:
        print("[STOPPED] Rerun the train command to resume.", flush=True)


if __name__ == "__main__":
    main()
