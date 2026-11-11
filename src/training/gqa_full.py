"""Matched GQA matrix with atomic, deterministic mid-epoch training resume."""

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from src.datasets.gqa_full import atomic_write, digest, pin_json
from src.knowledge.transe.core import sha256
from src.models.penet import PENetPredicateClassifier
from src.training.gqa_pilot import collate, cross_entropy_sum, forward
from src.training.gqa_tracking import RunTracker
from src.training.penet import prototype_losses
from tools.train_gqa_pilot import build_model, compact_metrics, evaluate
from tools.train_penet_pilot import state_hash

MODELS = ("mlp", "gat", "pe-ce", "pe-pl", "pe-plpr")


def setup_runtime(config):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(config["num_threads"])
    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")


def model_config(manifest):
    vocabulary = manifest["vocabulary"]
    return {
        "geometry_dim": 517,
        "embedding_dim": 32,
        "hidden_dim": 256,
        "num_object_classes": len(vocabulary["label_to_idx"]) + 1,
        "num_classes": len(vocabulary["predicate_to_idx"]) + 1,
        "semantic_dim": 256,
        "projection_dim": 512,
        "pe_dropout": 0.2,
        "gat": {"heads": 4, "layers": 2, "dropout": 0.1},
        "loss": {"k1": 10, "k2": 1, "gamma1": 1.0, "gamma2": 7.0, "weights": [1, 1, 1, 1]},
    }


def load_vectors(config, dimensions):
    path = Path(config["glove"])
    metadata = json.loads((path / "manifest.json").read_text())
    if metadata["manifest_sha256"] != sha256(Path(config["manifest"])):
        raise ValueError("GloVe vocabulary manifest mismatch")
    if sha256(path / "vectors.npz") != metadata["arrays_sha256"]:
        raise ValueError("GloVe vectors changed")
    with np.load(path / "vectors.npz", allow_pickle=False) as data:
        vectors = [torch.from_numpy(data[k].copy()) for k in ("objects", "predicates")]
    if [tuple(v.shape) for v in vectors] != [(dimensions["num_object_classes"], 300), (dimensions["num_classes"], 300)]:
        raise ValueError("GloVe shape mismatch")
    return vectors


def create_model(kind, dimensions, vectors):
    if kind in ("mlp", "gat"):
        return build_model(kind, dimensions)
    if kind not in MODELS:
        raise ValueError("Unknown model")
    return PENetPredicateClassifier(*vectors, semantic_dim=dimensions["semantic_dim"], dropout=dimensions["pe_dropout"])


def save_training(path, state, model, optimizer, generator):
    payload = {
        **state,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "sampler_rng": generator.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else [],
    }
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def train_run(
    kind, seed, train, val, vectors, experiment, directory, *, stop_requested=lambda: False, stop_after_batches=None
):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    identity = {"experiment_sha256": digest(experiment), "model": kind, "seed": seed}
    pin_json(directory / "identity.json", identity)
    tracker = RunTracker(experiment, identity, directory)
    status = "failed"
    try:
        result = _train_run(
            kind,
            seed,
            train,
            val,
            vectors,
            experiment,
            directory,
            tracker=tracker,
            stop_requested=stop_requested,
            stop_after_batches=stop_after_batches,
        )
        if result is not None:
            tracker.complete(result)
        status = "completed" if result is not None else "interrupted"
        return result
    finally:
        tracker.finish(status)


def update_early_stopping(state, score, settings):
    """Patience counts consecutive validations without improvement beyond min_delta."""
    if not settings.get("enabled", False):
        return
    if score > state["stopping_best"] + settings["min_delta"]:
        state["stopping_best"], state["bad_epochs"] = score, 0
    else:
        state["bad_epochs"] += 1
    state["stopped_early"] = state["bad_epochs"] >= settings["patience"]


def publish_best(directory, state, dimensions):
    """last.pt is authoritative; this derived export can be rebuilt after a crash."""
    if state["best_model"] is None:
        return
    best = Path(directory) / "best.pt"
    with best.with_suffix(".pt.tmp").open("wb") as f:
        torch.save(
            {
                "model": state["best_model"],
                "identity": state["identity"],
                "dimensions": dimensions,
                "epoch": state["best_epoch"],
                "score": state["best_score"],
            },
            f,
        )
        f.flush()
        os.fsync(f.fileno())
    os.replace(best.with_suffix(".pt.tmp"), best)


def _train_run(kind, seed, train, val, vectors, experiment, directory, *, tracker, stop_requested, stop_after_batches):
    """Resume automatically. Diagnostic stop count is per invocation, not experiment budget."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    identity = {"experiment_sha256": digest(experiment), "model": kind, "seed": seed}
    pin_json(directory / "identity.json", identity)
    config, dimensions = experiment["config"], experiment["dimensions"]
    device = config["device"]
    last, result_path = directory / "last.pt", directory / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result["identity"] != identity or sha256(directory / "best.pt") != result["checkpoint_sha256"]:
            raise ValueError("Completed run changed")
        print(f"[SKIP] verified complete {kind} seed={seed}", flush=True)
        return result
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    model = create_model(kind, dimensions, vectors).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    state = {
        "identity": identity,
        "initial_state_sha256": state_hash(model),
        "epoch": 1,
        "order": None,
        "next_index": 0,
        "history": [],
        "best_score": -1.0,
        "best_epoch": None,
        "best_model": None,
        "total_batches": 0,
        "train_seconds": 0.0,
        "bad_epochs": 0,
        "stopping_best": -1.0,
        "stopped_early": False,
    }
    if last.exists():
        saved = torch.load(last, map_location="cpu", weights_only=True)
        if saved["identity"] != identity:
            raise ValueError("Resume identity mismatch; use a fresh experiment")
        model.load_state_dict(saved.pop("model"))
        optimizer.load_state_dict(saved.pop("optimizer"))
        generator.set_state(saved.pop("sampler_rng"))
        torch.set_rng_state(saved.pop("torch_rng"))
        cuda_rng = saved.pop("cuda_rng")
        if device == "cuda":
            torch.cuda.set_rng_state_all(cuda_rng)
        state = saved
        publish_best(directory, state, dimensions)
        print(f"[RESUME] {kind} seed={seed} epoch={state['epoch']} next_index={state['next_index']}", flush=True)
    else:
        save_training(last, state, model, optimizer, generator)
    tracker.replay(state["history"])
    this_invocation = 0
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    while state["epoch"] <= config["epochs"] and not state["stopped_early"]:
        if state["order"] is None:
            state.update(
                order=torch.randperm(len(train), generator=generator).tolist(),
                next_index=0,
                loss_sums={k: 0.0 for k in ("ce", "pl_euclidean", "pr_cosine", "pr_euclidean", "total")},
                pairs=0,
                steps=0,
                empty_batches=0,
                max_grad_norm=0.0,
                trace="",
            )
        model.train()
        while state["next_index"] < len(train):
            if stop_requested():
                save_training(last, state, model, optimizer, generator)
                return None
            start = time.perf_counter()
            begin = state["next_index"]
            indices = state["order"][begin : begin + config["train_batch_size"]]
            graphs = [train[i] for i in indices]
            batch, _ = collate(graphs, dimensions["num_classes"], generator)
            h = hashlib.sha256(state["trace"].encode())
            h.update(json.dumps([g.image_id for g in graphs]).encode())
            h.update(batch["pairs"].numpy().tobytes())
            h.update(batch["targets"].numpy().tobytes())
            state["trace"] = h.hexdigest()
            count = len(batch["targets"])
            if count:
                optimizer.zero_grad(set_to_none=True)
                if kind.startswith("pe-"):
                    logits, relations, prototypes = model(
                        batch["x"].to(device),
                        batch["pairs"].to(device),
                        batch["object_labels"].to(device),
                        return_embeddings=True,
                    )
                    losses = prototype_losses(logits, relations, prototypes, batch["targets"].to(device), mode=kind[3:])
                else:
                    logits = forward(model, kind, batch, device)
                    loss = cross_entropy_sum(logits, batch["targets"].to(device)) / count
                    losses = {"ce": loss, "total": loss}
                if not torch.isfinite(losses["total"]):
                    raise RuntimeError("Non-finite loss")
                losses["total"].backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config["gradient_clip_norm"], error_if_nonfinite=True
                )
                optimizer.step()
                for key, value in losses.items():
                    state["loss_sums"][key] += float(value.detach()) * count
                state["max_grad_norm"] = max(state["max_grad_norm"], float(norm))
                state["pairs"] += count
                state["steps"] += 1
            else:
                state["empty_batches"] += 1
            state["next_index"] += len(indices)
            state["total_batches"] += 1
            state["train_seconds"] += time.perf_counter() - start
            this_invocation += 1
            stopping = stop_requested() or (stop_after_batches is not None and this_invocation >= stop_after_batches)
            if state["total_batches"] % config["checkpoint_every_batches"] == 0 or stopping:
                save_training(last, state, model, optimizer, generator)
                print(
                    f"[SAVE] {kind}/{seed} epoch={state['epoch']} images={state['next_index']}/{len(train)}", flush=True
                )
            if stopping:
                return None
        if not state["pairs"]:
            raise ValueError("No training pairs in epoch")
        eval_config = {**dimensions, "batch_size": config["val_batch_size"]}
        validation = compact_metrics(
            evaluate(model, "penet" if kind.startswith("pe-") else kind, val, eval_config, device)
        )
        if validation["mR@50"] is None or not all(math_finite(v) for v in validation.values()):
            raise ValueError("Validation has no GT or non-finite metrics")
        state["history"].append(
            {
                "epoch": state["epoch"],
                "validation": validation,
                "train_losses": {k: v / state["pairs"] for k, v in state["loss_sums"].items()},
                "train_pairs": state["pairs"],
                "steps": state["steps"],
                "skipped_empty_batches": state["empty_batches"],
                "sampling_trace_sha256": state["trace"],
                "max_gradient_norm_before_clip": state["max_grad_norm"],
            }
        )
        improved = validation["mR@50"] > state["best_score"]
        if improved:
            state["best_score"], state["best_epoch"] = validation["mR@50"], state["epoch"]
            state["best_model"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        update_early_stopping(state, validation["mR@50"], config.get("early_stopping", {}))
        state["history"][-1].update(
            best_score=state["best_score"],
            best_epoch=state["best_epoch"],
            early_stopping_bad_epochs=state["bad_epochs"],
        )
        print(
            f"[EPOCH] {kind}/{seed} {state['epoch']}/{config['epochs']} R50={validation['R@50']:.6f} mR50={validation['mR@50']:.6f}",
            flush=True,
        )
        state["epoch"] += 1
        state["order"] = None
        save_training(last, state, model, optimizer, generator)
        if improved:
            publish_best(directory, state, dimensions)
        atomic_write(directory / "history.json", state["history"])
        tracker.replay(state["history"][-1:])
    if stop_requested():
        return None
    model.load_state_dict(state["best_model"])
    final = evaluate(
        model,
        "penet" if kind.startswith("pe-") else kind,
        val,
        {**dimensions, "batch_size": config["val_batch_size"]},
        device,
        include_ng=True,
    )
    if final["gc"]["mR@50"] != state["best_score"]:
        raise AssertionError("Best checkpoint reload metric differs")
    best = directory / "best.pt"
    result = {
        "status": "PASS",
        "scope": config["scope"],
        "identity": identity,
        "initial_state_sha256": state["initial_state_sha256"],
        "selected_epoch": state["best_epoch"],
        "epochs_completed": len(state["history"]),
        "stopping_reason": "early_stopping" if state["stopped_early"] else "max_epochs",
        "early_stopping_bad_epochs": state["bad_epochs"],
        "history": state["history"],
        "validation": final,
        "checkpoint_sha256": sha256(best),
        "parameters": sum(p.numel() for p in model.parameters()),
        "train_images": len(train),
        "val_images": len(val),
        "train_seconds_excluding_validation_and_checkpoint_io": state["train_seconds"],
        "peak_cuda_allocated_bytes_this_invocation": torch.cuda.max_memory_allocated() if device == "cuda" else None,
        "test_used": False,
    }
    atomic_write(result_path, result)
    return result


def math_finite(value):
    import math

    return value is not None and math.isfinite(value)


def summarize_matrix(experiment, directory):
    """Only publish a complete matrix; report sample standard deviation (ddof=1)."""
    config = experiment["config"]
    rows = {}
    for model in config["models"]:
        rows[model] = []
        for seed in config["seeds"]:
            run = Path(directory) / model / str(seed)
            result = json.loads((run / "result.json").read_text())
            if result["identity"] != {"experiment_sha256": digest(experiment), "model": model, "seed": seed}:
                raise ValueError("Mixed experiment results")
            if result["status"] != "PASS" or sha256(run / "best.pt") != result["checkpoint_sha256"]:
                raise ValueError("Invalid result/checkpoint")
            rows[model].append(result)
    for i, seed in enumerate(config["seeds"]):
        traces = [[h["sampling_trace_sha256"] for h in rows[m][i]["history"]] for m in config["models"]]
        longest = max(traces, key=len)
        if any(t != longest[: len(t)] for t in traces):
            raise AssertionError(f"Sampling not matched for seed {seed}")
        initial = [rows[m][i]["initial_state_sha256"] for m in config["models"] if m.startswith("pe-")]
        if len(set(initial)) != 1:
            raise AssertionError("PE initial states differ")
    summaries = {}
    for model, results in rows.items():
        summaries[model] = {}
        for protocol, prefix in (("gc", ""), ("ng", "ng-")):
            for name in ("R@50", "R@100", "mR@50", "mR@100"):
                key = prefix + name
                values = [r["validation"][protocol][key] for r in results]
                summaries[model][key] = {
                    "values": values,
                    "mean": float(np.mean(values)),
                    "std_sample": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                }
    output = {
        "status": "PASS",
        "scope": config["scope"],
        "experiment_sha256": digest(experiment),
        "seeds": config["seeds"],
        "models": summaries,
        "matched_sampling": True,
        "sampling_comparison": "identical common epoch prefixes; early stopping can change run lengths",
        "training_lengths": {
            model: [
                {
                    "seed": seed,
                    "epochs_completed": r["epochs_completed"],
                    "stopping_reason": r["stopping_reason"],
                    "selected_epoch": r["selected_epoch"],
                }
                for seed, r in zip(config["seeds"], results, strict=True)
            ]
            for model, results in rows.items()
        },
        "matched_pe_initialization": True,
        "units": "fractions, multiply by 100 for percent",
        "test_used": False,
    }
    atomic_write(Path(directory) / "summary.json", output)
    return output
