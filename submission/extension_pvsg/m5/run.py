"""Frozen held-out evaluation of validation-selected PVSG checkpoints."""

import argparse
import fcntl
import os
from pathlib import Path

import torch

from extension_pvsg.contracts import read_json
from extension_pvsg.evaluation import evaluate
from extension_pvsg.intervals import window_bounds
from extension_pvsg.m2.download import sha256
from extension_pvsg.m2.preflight import verify_locks
from extension_pvsg.m3.common import PROJECT, write_json
from extension_pvsg.m3.data import Store
from extension_pvsg.m3.data import collate as collate_m3
from extension_pvsg.m3.data import to_device
from extension_pvsg.m3.decoding import Accumulator
from extension_pvsg.m3.engine import predict_batch
from extension_pvsg.m3.models import RelationModel as M3Model
from extension_pvsg.m3.preflight import check_features, check_runtime
from extension_pvsg.m4.common import verify_parent, verify_self
from extension_pvsg.m4.data import collate as collate_m4
from extension_pvsg.m4.models import RelationModel as M4Model
from extension_pvsg.min50_followup.run import CASES
from extension_pvsg.validate_m0 import canonical_sha256
from extension_pvsg.m5.report import diagnostics, summarize

OUT = Path("reports/pvsg-trd-m5-v1")
M3 = ("mlp", "static_gat", "st_gat")
M4 = ("pvsg_conv_k5_l3", "st_no_temporal", "st_no_spatial", "st_no_edges", "mlp_conv_k3_l2")
SEEDS = (17, 42, 2026)


def roster():
    for family, models, folder in (("m3", M3, "m3-150-v1"), ("m4", M4, "m4-150-bs32-v1")):
        for model in models:
            for seed in SEEDS:
                yield "baseline", family, model, seed, Path(f"reports/pvsg-trd-{folder}")
    for model, seeds in CASES.items():
        for seed in seeds:
            yield "min50", "m3" if model in M3 else "m4", model, seed, Path("reports/pvsg-trd-min50-followup-v1")


def inventory():
    rows, pending = [], []
    for group, family, model, seed, root in roster():
        folder = root / model / f"seed-{seed}"
        if not (folder / "complete.json").exists():
            pending.append(str(folder))
            continue
        complete = read_json(folder / "complete.json")
        run = read_json(folder / "run.json")
        ident = run["identity"]
        history = read_json(folder / "history.json")
        best = max(history, key=lambda r: r["selection_key"])
        weight = folder / "best.pt"
        if (
            complete["test_evaluated"]
            or complete["signature"] != canonical_sha256(ident)
            or run["signature"] != complete["signature"]
            or ident["model"] != model
            or ident["seed"] != seed
            or complete["best_key"] != best["selection_key"]
            or sha256(weight) != complete["best_checkpoint_sha256"]
            or sha256(folder / "history.json") != complete["history_sha256"]
        ):
            raise ValueError(f"Invalid baseline identity or best checkpoint: {folder}")
        rows.append(
            dict(
                group=group,
                family=family,
                model=model,
                seed=seed,
                checkpoint=str(weight),
                sha256=complete["best_checkpoint_sha256"],
                signature=complete["signature"],
                epoch=best["epoch"],
                validation_key=best["selection_key"],
            )
        )
    return rows, pending


def network(row):
    ckpt = torch.load(row["checkpoint"], map_location="cpu", weights_only=True)
    cfg = ckpt["identity"]["config"]
    if (
        ckpt["signature"] != row["signature"]
        or canonical_sha256(ckpt["identity"]) != row["signature"]
        or ckpt["epoch"] != row["epoch"]
        or ckpt["best_key"] != row["validation_key"]
        or cfg["decoder_threshold"] != 0.5
    ):
        raise ValueError("Checkpoint differs from frozen selection")
    factory = M3Model if row["family"] == "m3" else M4Model
    model = factory(row["model"], cfg)
    model.load_state_dict(ckpt["model"], strict=True)
    return model.eval(), cfg, ckpt["identity"]["feature_summary_sha256"]


@torch.inference_mode()
def predict_video(model, family, store, vid, cfg, device):
    video = store.videos[vid]
    acc = Accumulator(video)
    collate = collate_m3 if family == "m3" else collate_m4
    for start, stop in window_bounds(video["num_frames"], cfg["window_frames"], cfg["validation_stride"]):
        sample = store.window(vid, start, stop - start, supervision=False)
        batch = to_device(collate([sample]), device)
        scores = predict_batch(model, batch, cfg["pair_chunk_size"], cfg["precision"])
        acc.add(sample["graph"], scores, start)
    return acc.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "evaluate"))
    args = parser.parse_args()
    os.chdir(PROJECT)
    verify_locks()
    check_runtime()
    verify_parent()
    verify_self()
    rows, pending = inventory()
    print(f"Validation-selected best checkpoints: {len(rows)}/31. Pending: {pending}", flush=True)
    _, cfg, digest = network(rows[0])
    if check_features(cfg) != digest:
        raise ValueError("Feature cache identity differs from training")
    store = Store(cfg["dataset_cache"], cfg["feature_cache"])
    ids = sorted(store.manifest["outer_split"]["heldout_ids"])
    if len(set(ids)) != 62 or len(ids) != 62 or set(ids) & set(store.manifest["outer_split"]["train_ids"]):
        raise ValueError("Invalid held-out cohort")
    plan = dict(
        runs=rows,
        heldout_ids=ids,
        features=digest,
        split_sha256=sha256(store.cache_root / "dataset.lock.json"),
        evaluate_code=sha256(__file__),
        report_code=sha256(Path(__file__).with_name("report.py")),
        batch_windows=1,
        case_study="static_gat seed 17; first five hits and misses in ID/key order",
    )
    if args.action == "check":
        for row in rows:
            _, _, other = network(row)
            if other != digest:
                raise ValueError("Checkpoint feature cache mismatch")
        return
    if pending:
        raise SystemExit("Finish the seven min50 follow-ups before opening held-out evaluation.")
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        freeze = OUT / "frozen.json"
        if freeze.exists() and read_json(freeze) != plan:
            raise ValueError("Existing frozen plan differs")
        write_json(freeze, plan)
        signature = canonical_sha256(plan)
        device = torch.device("cuda:0")
        if not torch.cuda.is_bf16_supported():
            raise ValueError("CUDA BF16 required")
        results = []
        for row in rows:
            dest = OUT / row["group"] / row["model"] / f"seed-{row['seed']}.json"
            if dest.exists():
                result = read_json(dest)
                if result["freeze"] != signature or result["run"] != row:
                    raise ValueError(f"Incompatible completed result: {dest}")
            else:
                model, settings, other = network(row)
                if other != digest:
                    raise ValueError("Checkpoint feature cache mismatch")
                model.to(device)
                predictions = [predict_video(model, row["family"], store, vid, settings, device) for vid in ids]
                metrics = evaluate(
                    [store.videos[v] for v in ids], [store.targets[v] for v in ids], predictions, expected_video_ids=ids
                )
                details = diagnostics(store, ids, predictions, metrics, row, OUT)
                result = dict(freeze=signature, run=row, metrics=metrics, details=details)
                write_json(dest, result)
                del model
            results.append(result)
            print(dest, flush=True)
        summarize(results, OUT)


if __name__ == "__main__":
    main()
