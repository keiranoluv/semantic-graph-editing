"""One main training run: mandatory online WandB, internal-val, epoch resume."""

import os
import time
from pathlib import Path

import numpy as np
import torch
from extension_pvsg.contracts import read_json
from extension_pvsg.m2.download import sha256
from extension_pvsg.m3.common import identity, seed_all, write_json
from extension_pvsg.m3.data import FitSampler, Store, Windows, collate, to_device
from extension_pvsg.m3.engine import (
    backward_microbatch,
    diagnostics,
    restore_checkpoint,
    save_checkpoint,
    selection_key,
    should_stop,
)
from extension_pvsg.m3.models import RelationModel
from extension_pvsg.m3.validation import validate
from extension_pvsg.validate_m0 import canonical_sha256
from torch.utils.data import DataLoader


def tracking(cfg, kind, seed, out, run_id, run_identity):
    import wandb

    if os.environ.get("WANDB_MODE", "online") != "online":
        raise ValueError("Main training requires WANDB_MODE=online")
    run = wandb.init(
        project=cfg["wandb_project"],
        group=cfg["experiment_id"],
        name=f"{kind}-seed{seed}",
        id=run_id,
        resume="must" if (out / "last.pt").exists() else "allow",
        mode="online",
        dir=str(out),
        config=run_identity,
        settings=wandb.Settings(init_timeout=120),
    )
    if run is None or run.settings.mode != "online" or not run.url:
        raise RuntimeError("Online WandB initialization failed; training will not start")
    run.log({"preflight/online": 1, "preflight/seed": seed})
    return run


def train(cfg, kind, seed, batch_size, feature_sha, device):
    seed_all(seed)
    cuda = torch.device(device).type == "cuda"
    effective = cfg["effective_batch_windows"]
    out = Path(cfg["output_root"]) / kind / f"seed-{seed}"
    out.mkdir(parents=True, exist_ok=True)
    run_identity = {
        **identity(cfg, feature_sha),
        "model": kind,
        "seed": seed,
        "batch_size": batch_size,
        "batch_policy": "explicit_fixed_batch",
    }
    sig = canonical_sha256(run_identity)
    if (out / "complete.json").exists():
        completed = read_json(out / "complete.json")
        if completed["signature"] != sig or sha256(out / "best.pt") != completed["best_checkpoint_sha256"]:
            raise ValueError("Completed run differs from current experiment or checkpoint")
        print(f"Verified completed run, skipping: {out}", flush=True)
        return
    run_file = out / "run.json"
    if run_file.exists():
        run_info = read_json(run_file)
        if run_info["signature"] != sig:
            raise ValueError("Existing run identity mismatch; use a new experiment version")
    else:
        import wandb

        run_info = dict(signature=sig, wandb_id=wandb.util.generate_id(), identity=run_identity)
        write_json(run_file, run_info)
    tracker = tracking(cfg, kind, seed, out, run_info["wandb_id"], run_identity)
    model = RelationModel(kind, cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg["lr_plateau_factor"],
        patience=cfg["lr_plateau_patience"],
        min_lr=cfg["min_learning_rate"],
    )
    start, bad, best, history = 0, 0, (-1.0, -1.0), []
    if (out / "last.pt").exists():
        ckpt = restore_checkpoint(out / "last.pt", model, optimizer, scheduler, sig, device)
        start, bad, best, history = ckpt["epoch"], ckpt["bad_epochs"], tuple(ckpt["best_key"]), ckpt["history"]
    store = Store(cfg["dataset_cache"], cfg["feature_cache"], cfg["worker_video_cache"])
    sampler = FitSampler(store, cfg["window_frames"], cfg["windows_per_fit_video_per_epoch"], seed)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        Windows(store, cfg["window_frames"]),
        sampler=sampler,
        batch_size=batch_size,
        collate_fn=collate,
        num_workers=cfg["loader_workers"],
        pin_memory=cuda,
        persistent_workers=cfg["loader_workers"] > 0,
        prefetch_factor=cfg["prefetch_factor"] if cfg["loader_workers"] else None,
        multiprocessing_context="spawn" if cfg["loader_workers"] else None,
        generator=generator,
    )
    tracker.config.update(
        {
            "parameters": sum(p.numel() for p in model.parameters()),
            "gpu": torch.cuda.get_device_name(device) if cuda else "CPU integration fixture",
            "effective_batch_windows": effective,
        },
        allow_val_change=False,
    )
    try:
        for epoch in range(start + 1, cfg["max_epochs"] + 1):
            if should_stop(start, bad, cfg):
                break
            started = time.monotonic()
            model.train()
            sampler.epoch = epoch
            totals, processed, group_processed, optimizer_steps = np.zeros(9), 0, 0, 0
            optimizer.zero_grad(set_to_none=True)
            if cuda:
                torch.cuda.reset_peak_memory_stats(device)
            gradient_norms = []
            group_windows = min(effective, len(sampler))
            loader_wait_seconds = 0.0
            wait_start = time.monotonic()
            for cpu_batch in loader:
                loader_wait_seconds += time.monotonic() - wait_start
                batch = to_device(cpu_batch, device)
                totals += backward_microbatch(
                    model,
                    batch,
                    chunk_size=cfg["pair_chunk_size"],
                    group_windows=group_windows,
                    precision=cfg["precision"],
                )
                count = batch["sample_count"]
                group_processed += count
                processed += count
                if group_processed == group_windows:
                    norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg["gradient_clip_norm"], error_if_nonfinite=True
                    )
                    gradient_norms.append(float(norm))
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1
                    group_processed = 0
                    group_windows = min(effective, len(sampler) - processed)
                del batch
                wait_start = time.monotonic()
            if processed != len(sampler) or group_processed:
                raise ValueError("Training sampler/gradient accumulation accounting mismatch")
            if cuda:
                torch.cuda.synchronize(device)
            train_seconds = time.monotonic() - started
            report = validate(model, store, cfg, device, batch_size)
            key = selection_key(report)
            improved = key > best
            best, bad = (key, 0) if improved else (best, bad + 1)
            scheduler.step(report["validation_bce"])
            train_stats = diagnostics(totals, processed)
            row = dict(
                epoch=epoch,
                train=train_stats,
                validation_bce=report["validation_bce"],
                selection_key=list(key),
                learning_rate=optimizer.param_groups[0]["lr"],
                gradient_norm_mean=float(np.mean(gradient_norms)),
                optimizer_steps=optimizer_steps,
                train_seconds=train_seconds,
                loader_wait_seconds=loader_wait_seconds,
                epoch_seconds=time.monotonic() - started,
                windows_per_second=processed / train_seconds,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if cuda else 0,
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if cuda else 0,
                all_negative_prediction_warning=report["predicted_tuples"] == 0,
            )
            history.append(row)
            write_json(out / "history.json", history)
            write_json(out / f"validation/epoch-{epoch:03d}.json", report)
            payload = dict(
                signature=sig,
                epoch=epoch,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                best_key=list(best),
                bad_epochs=bad,
                history=history,
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if cuda else [],
                wandb_id=run_info["wandb_id"],
                identity=run_identity,
            )
            if improved:
                save_checkpoint(out / "best.pt", payload)
                write_json(out / "best_validation.json", report)
            save_checkpoint(out / "last.pt", payload)
            logs = {f"train/{k}": v for k, v in train_stats.items()}
            logs.update(
                {
                    "epoch": epoch,
                    "validation/BCE": report["validation_bce"],
                    "optimizer/lr": row["learning_rate"],
                    "optimizer/gradient_norm": row["gradient_norm_mean"],
                    "resources/peak_allocated_GiB": row["peak_allocated_bytes"] / 2**30,
                    "resources/peak_reserved_GiB": row["peak_reserved_bytes"] / 2**30,
                    "resources/windows_per_second": row["windows_per_second"],
                    "resources/epoch_seconds": row["epoch_seconds"],
                    "resources/loader_wait_seconds": loader_wait_seconds,
                    "validation/mean_tiou_at_100": report["diagnostics"]["mean_tiou_at_100"],
                    "validation/all_negative_warning": int(row["all_negative_prediction_warning"]),
                }
            )
            for tau, metrics in report["metrics"].items():
                for k, values in metrics.items():
                    for name in ("R", "mR"):
                        logs[f"validation/{name}{k}_tIoU{tau}"] = values[name]
            tracker.log(logs)
            print(
                f"{kind} seed={seed} epoch={epoch}: loss={train_stats['loss']:.5f}, val mR/R50={key}, BS={batch_size}",
                flush=True,
            )
            if should_stop(epoch, bad, cfg):
                break
        completed = dict(
            signature=sig,
            epochs=history[-1]["epoch"],
            best_key=list(best),
            wandb_url=tracker.url,
            best_checkpoint_sha256=sha256(out / "best.pt"),
            history_sha256=sha256(out / "history.json"),
            test_evaluated=False,
            early_stopped=should_stop(history[-1]["epoch"], bad, cfg),
        )
        tracker.summary.update(completed)
        tracker.finish(exit_code=0)
        write_json(out / "complete.json", completed)
    except BaseException:
        tracker.finish(exit_code=1)
        raise
