"""CPU preflight/export; user-run sequential four-arm CUDA pilot, no test stage."""
import argparse
import gc
import json
from pathlib import Path

import torch

from src.datasets.gqa_full import atomic_write
from tools.emnlp2020_graph_modification_v1.dataset import read, sha256, write
from tools.magicbrush_experiment_v3.cache import build_cache
from tools.magicbrush_experiment_v3.pipeline import source_hashes as parent_hashes
from tools.magicbrush_experiment_v3.pipeline import versions
from tools.magicbrush_lexical_pilot_v1.analysis import diagnose
from tools.prepare_magicbrush import ROOT, digest
from tools.run_gqa_full import exclusive

from .data import export, prepare
from .features import GroundingStore, cache_inputs
from .training import train_one

CONFIG = ROOT / "configs/level2/magicbrush-easy-copy-v1.json"
FILES = ("best.pt", "last.pt", "result.json", "predictions-val.json", "diagnostics.json")


def source_hashes():
    paths = [
        *Path(__file__).parent.glob("*.py"),
        *ROOT.glob("tools/magicbrush_lexical_pilot_v1/*.py"),
        ROOT / "tools/magicbrush_compact_v1/build.py",
        ROOT / "tools/magicbrush_compact_v1/projection.py",
        ROOT / "scripts/run_magicbrush_easy_copy_v1.sh",
    ]
    return {**parent_hashes(), **{str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}}


def preflight():
    cfg = read(CONFIG)
    lock = {"config_sha256": sha256(CONFIG), "source_sha256": source_hashes(), "versions": versions()}
    if read(CONFIG.with_suffix(".lock.json")) != lock:
        raise ValueError("Easy/copy release changed")
    if cfg["schema"] != "magicbrush-easy-copy-v1" or cfg["test_used"]:
        raise ValueError("Wrong experiment scope")
    for key, expected in {
        "output": "runs/level2/magicbrush-easy-copy-v1",
        "dataset_output": "data/processed/level2/magicbrush-easy-copy-v1",
        "reports": "reports/level2/magicbrush-easy-copy-v1",
        "text_cache": "data/processed/level2/magicbrush-easy-copy-v1/clip-trainval",
    }.items():
        if cfg[key] != expected:
            raise ValueError("Output is not isolated")
    examples, vocab, records, sidecar, report = prepare(cfg)
    identity = {
        "schema": cfg["schema"],
        "config": cfg,
        "release_lock_sha256": sha256(CONFIG.with_suffix(".lock.json")),
        "parent_identity_sha256": report["parent_identity_sha256"],
        "dataset_records_sha256": digest(records),
        "vocabulary": vocab,
        "vocabulary_sha256": digest(vocab),
        "candidate_scope": "per-example normalized contiguous instruction spans only; CLIP frozen",
    }
    export(cfg, identity, records, sidecar, report)
    return cfg, examples, identity, report


def completed(directory, run_id):
    receipt = read(directory / "completed-receipt.json")
    if receipt["identity"] != run_id or set(receipt["files"]) != set(FILES):
        raise ValueError("Completion identity/files changed")
    if any(sha256(directory / name) != h for name, h in receipt["files"].items()):
        raise ValueError("Completed artifacts modified")
    return read(directory / "result.json"), read(directory / "diagnostics.json")


def run_queue(cfg, examples, store, identity, *, device="cuda"):
    output = ROOT / cfg["output"]
    results = {}
    for arm in cfg["queue"]:
        variant, kind = arm["variant"], arm["model"]
        local = {**cfg, "use_copy": variant == "copy"}
        local_identity = {**identity, "variant": variant}
        directory = output / variant / kind / str(cfg["seed"])
        run_id = {"experiment_sha256": digest(local_identity), "model": kind, "seed": cfg["seed"]}
        atomic_write(output / "progress.json", {"stage": "train_validate", **arm, "seed": cfg["seed"]})
        if (directory / "completed-receipt.json").exists():
            result, diagnostic = completed(directory, run_id)
        else:
            result = train_one(examples, store, local, local_identity, kind, cfg["seed"], directory, device=device)
            rows = read(directory / "predictions-val.json")
            diagnostic = diagnose(examples["val"], rows)
            write(directory / "diagnostics.json", diagnostic)
            write(
                directory / "completed-receipt.json",
                {"identity": run_id, "files": {n: sha256(directory / n) for n in FILES}},
            )
        results[variant + "/" + kind] = {
            "status": result["status"],
            "epoch": result["selected_epoch"],
            "epochs_completed": result["epochs_completed"],
            "stopping_reason": result["stopping_reason"],
            "validation": result["validation"],
            "raw_validation": result["raw_validation"],
            "diagnostics": diagnostic,
        }
        # Weak validation does not suppress the paired control. Nonfinite losses,
        # invalid artifacts or other runtime exceptions do stop the entire queue.
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    summary = {
        "identity_sha256": digest(identity),
        "models": results,
        "scope": "Four separately trained seed-42 arms on identical new labels and all 67 validation pairs",
        "comparison": {
            m: {
                "no_copy_f1": results["no_copy/" + m]["validation"]["micro"]["f1"],
                "copy_f1": results["copy/" + m]["validation"]["micro"]["f1"],
            }
            for m in ("mlp", "gnn")
        },
        "test_used": False,
        "main_three_seed_experiment": False,
    }
    write(output / "summary.json", summary)
    atomic_write(output / "progress.json", {"stage": "complete", "summary": str(output / "summary.json")})
    print(json.dumps({"stage": "complete", "comparison": summary["comparison"]}), flush=True)
    return summary


def execute(cfg, examples, identity):
    output = ROOT / cfg["output"]
    with exclusive(ROOT / "runs/level2/.gqa-kg-training-device"), exclusive(output):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; CPU preflight does not start training")
        torch.set_num_threads(1)
        torch.backends.cudnn.benchmark = False
        write(output / "experiment.json", identity)
        items = examples["train"] + examples["val"]
        atomic_write(output / "progress.json", {"stage": "cache_frozen_text_and_instruction_spans"})
        base, meta = build_cache(
            items + cache_inputs(items), cfg, identity, ROOT / cfg["text_cache"], ROOT, scope="trainval"
        )
        trained_identity = {**identity, "frozen_vectors_sha256": meta["vectors_sha256"]}
        write(output / "training-identity.json", trained_identity)
        store = GroundingStore(base, identity["vocabulary"])
        try:
            return run_queue(cfg, examples, store, trained_identity)
        finally:
            gc.collect()
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "run"), default="preflight")
    args = parser.parse_args()
    cfg, examples, identity, report = preflight()
    print(
        json.dumps(
            {
                "stage": "preflight",
                "train": len(examples["train"]),
                "val": len(examples["val"]),
                "copy_coverage": report["copy_name_coverage"],
                "test_used": False,
            }
        ),
        flush=True,
    )
    if args.stage == "run":
        execute(cfg, examples, identity)


if __name__ == "__main__":
    main()
