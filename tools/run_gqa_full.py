"""Preflight, resumable cache, matched 15-run training, smoke proof and aggregation."""

import argparse
import copy
import fcntl
import gc
import importlib.metadata
import json
import shutil
import signal
import sys
from contextlib import contextmanager
from pathlib import Path

import torch

from src.datasets.gqa_full import (
    GQAFullDataset,
    atomic_write,
    cache_identity,
    cohort_for,
    digest,
    pin_json,
    prepare_cache,
    verify_cache,
)
from src.knowledge.transe.core import sha256
from src.training.gqa_full import MODELS, load_vectors, model_config, setup_runtime, summarize_matrix, train_run


def load_config(path):
    config = json.loads(Path(path).read_text())
    required = {
        "schema_version",
        "scope",
        "root",
        "manifest",
        "weights",
        "glove",
        "cache",
        "output",
        "models",
        "seeds",
        "epochs",
        "train_batch_size",
        "val_batch_size",
        "lr",
        "weight_decay",
        "gradient_clip_norm",
        "checkpoint_every_batches",
        "cache_shard_size",
        "clip_batch_size",
        "device",
        "num_threads",
    }
    if config.get("schema_version") == 2:
        required |= {"early_stopping", "wandb"}
    if set(config) != required or config["schema_version"] not in (1, 2) or config["scope"] != "full":
        raise ValueError("Expected full experiment schema v1/v2 with exactly the documented fields")
    if config["schema_version"] == 2:
        early, tracking = config["early_stopping"], config["wandb"]
        if (
            set(early) != {"enabled", "patience", "min_delta"}
            or type(early["enabled"]) is not bool
            or type(early["patience"]) is not int
            or early["patience"] <= 0
            or not isinstance(early["min_delta"], (int, float))
            or early["min_delta"] < 0
            or not torch.isfinite(torch.tensor(early["min_delta"]))
        ):
            raise ValueError("Invalid early stopping configuration")
        if (
            set(tracking) != {"mode", "entity", "project"}
            or tracking["mode"] not in ("online", "offline", "disabled")
            or any(not isinstance(tracking[k], str) or not tracking[k].strip() for k in ("entity", "project"))
        ):
            raise ValueError("Invalid W&B configuration")
    if config["models"] != list(MODELS) or config["seeds"] != [42, 123, 3407]:
        raise ValueError("Full matrix must contain all five models and seeds 42/123/3407")
    for key in (
        "epochs",
        "train_batch_size",
        "val_batch_size",
        "checkpoint_every_batches",
        "cache_shard_size",
        "clip_batch_size",
        "num_threads",
    ):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"Invalid positive integer: {key}")
    if (
        config["device"] not in ("cpu", "cuda")
        or config["lr"] <= 0
        or config["weight_decay"] < 0
        or config["gradient_clip_norm"] <= 0
    ):
        raise ValueError("Invalid runtime/optimization configuration")
    if not all(torch.isfinite(torch.tensor(config[k])) for k in ("lr", "weight_decay", "gradient_clip_norm")):
        raise ValueError("Non-finite optimization configuration")
    return config


@contextmanager
def exclusive(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another process is writing {directory}") from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def preflight(config):
    manifest_path = Path(config["manifest"])
    manifest = json.loads(manifest_path.read_text())
    expected = "c953dc40d4b0a880ccd142f36670388ebffb7d7e2bb341d661b8b45781046c2a"
    if sha256(manifest_path) != expected:
        raise ValueError("This recipe requires the frozen GQA v1 manifest")
    cohort = cohort_for(config, manifest)
    if config["scope"] == "full" and [len(cohort[k]) for k in ("train", "val")] != [69942, 5000]:
        raise ValueError("Incomplete full split")
    annotation = Path(config["root"]) / "sceneGraphs/train_sceneGraphs.json"
    if sha256(annotation) != manifest["annotation_sha256"][annotation.name]:
        raise ValueError("Train annotation changed")
    if not (Path(config["root"]) / "archives/images.zip").is_file():
        raise FileNotFoundError("GQA image ZIP missing")
    dimensions = model_config(manifest)
    if dimensions["num_classes"] != 311 or dimensions["num_object_classes"] != 1700:
        raise ValueError("Wrong GQA taxonomy")
    load_vectors(config, dimensions)
    pilot = json.loads(Path("data/cache/gqa-pilot-clip-v1/manifest.json").read_text())
    if sha256(Path(config["weights"])) != pilot["weights_sha256"]:
        raise ValueError("CLIP weights differ from the matched pilot")
    setup_runtime(config)
    result = {
        "status": "READY",
        "scope": config["scope"],
        "train_images": len(cohort["train"]),
        "val_images": len(cohort["val"]),
        "runs": len(config["models"]) * len(config["seeds"]),
        "epochs_each": config["epochs"],
        "early_stopping": config.get("early_stopping", {"enabled": False}),
        "wandb": config.get("wandb", {"mode": "disabled"}),
        "device": config["device"],
        "gpu": torch.cuda.get_device_name() if config["device"] == "cuda" else None,
        "free_disk_bytes_workspace": shutil.disk_usage(Path.cwd()).free,
        "cache_complete": (Path(config["cache"]) / "manifest.json").exists(),
        "manifest_sha256": expected,
        "test_annotation_opened": False,
        "note": "Fixed shared starting recipe, not a convergence guarantee. No full training is started by preflight.",
    }
    print(json.dumps(result, indent=2), flush=True)
    return manifest, result


def experiment_identity(config, manifest):
    files = [
        *sorted(Path("src").rglob("*.py")),
        Path("tools/run_gqa_full.py"),
        Path("tools/train_gqa_pilot.py"),
        Path("tools/train_penet_pilot.py"),
    ]
    return {
        "schema_version": 1,
        "config": config,
        "dimensions": model_config(manifest),
        "manifest_sha256": sha256(Path(config["manifest"])),
        "cache_manifest_sha256": sha256(Path(config["cache"]) / "manifest.json"),
        "glove_manifest_sha256": sha256(Path(config["glove"]) / "manifest.json"),
        "code_sha256": {str(f): sha256(f) for f in files},
        "torch_version": str(torch.__version__),
        "wandb_version": importlib.metadata.version("wandb")
        if config.get("wandb", {}).get("mode", "disabled") != "disabled"
        else None,
        "gpu": torch.cuda.get_device_name() if config["device"] == "cuda" else None,
        "protocol": "unchanged GQA GC/NG reference IoU50; mR over all 310 predicates; uniform multi-positive CE",
        "sampling": "all positives + up to 3 backgrounds per positive; no-GT up to 32; empty batch no update",
        "selection": "maximum validation GC mR@50; earliest tie; optional patience/min_delta early stopping; no scheduler",
        "resume": "model+optimizer+CPU/CUDA/sampler RNG+order+offset+best+history; atomic last.pt",
        "test_used": False,
    }


def run_matrix(config, manifest, args, stopped):
    metadata = verify_cache(config["cache"], cache_identity(config, manifest, cohort_for(config, manifest)))
    experiment = experiment_identity(config, manifest)
    output = Path(config["output"])
    pin_json(output / "experiment.json", experiment)
    vectors = load_vectors(config, experiment["dimensions"])
    train, val = [GQAFullDataset(config["cache"], metadata, k) for k in ("train", "val")]
    try:
        for model in config["models"]:
            if args.model and args.model != model:
                continue
            for seed in config["seeds"]:
                if args.seed is not None and args.seed != seed:
                    continue
                if stopped():
                    return False
                result = train_run(
                    model,
                    seed,
                    train,
                    val,
                    vectors,
                    experiment,
                    output / model / str(seed),
                    stop_requested=stopped,
                    stop_after_batches=args.stop_after_batches,
                )
                gc.collect()
                if config["device"] == "cuda":
                    torch.cuda.empty_cache()
                if result is None:
                    print(
                        "[STOPPED] checkpoint saved; rerun the same command without diagnostic stop to resume",
                        flush=True,
                    )
                    return False
        if not args.model and args.seed is None:
            summarize_matrix(experiment, output)
        if args.stage == "smoke":
            # Independent interrupted copy must match the uninterrupted PE-PLPR run exactly.
            resumed = output / "resume-proof" / "pe-plpr" / "42"
            if not (resumed / "last.pt").exists():
                assert train_run("pe-plpr", 42, train, val, vectors, experiment, resumed, stop_after_batches=1) is None
            actual = train_run("pe-plpr", 42, train, val, vectors, experiment, resumed, stop_requested=stopped)
            if actual is None:
                return False
            expected = json.loads((output / "pe-plpr/42/result.json").read_text())
            for key in ("history", "validation", "initial_state_sha256", "selected_epoch"):
                if actual[key] != expected[key]:
                    raise AssertionError(f"Resume differs: {key}")
            a = torch.load(resumed / "last.pt", map_location="cpu", weights_only=True)
            b = torch.load(output / "pe-plpr/42/last.pt", map_location="cpu", weights_only=True)
            for key in a["model"]:
                torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
            proof = {
                "status": "PASS",
                "scope": "smoke, not main experiment",
                "interrupted_after_batches": 1,
                "exact_history_metrics_and_final_parameters": True,
                "experiment_sha256": digest(experiment),
            }
            atomic_write(output / "resume-proof.json", proof)
            print("[PASS] real-data interrupted/resumed PE-PLPR matches uninterrupted run exactly", flush=True)
    finally:
        train.close()
        val.close()
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/level2/gqa-full-v2.json"))
    parser.add_argument(
        "--stage", choices=("preflight", "cache", "train", "summary", "all", "smoke"), default="preflight"
    )
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--seed", type=int, choices=(42, 123, 3407))
    parser.add_argument(
        "--stop-after-batches", type=int, help="Diagnostic interruption; rerun without this option to finish"
    )
    args = parser.parse_args(argv)
    if args.stop_after_batches is not None and args.stop_after_batches <= 0:
        parser.error("stop-after-batches must be positive")
    if (args.model or args.seed is not None or args.stop_after_batches) and args.stage not in ("train", "all"):
        parser.error("Run filters/diagnostic stop apply only to train/all")
    config = load_config(args.config)
    if args.stage == "smoke":
        config = copy.deepcopy(config)
        config.update(
            scope="smoke",
            smoke_train_size=32,
            smoke_val_size=16,
            epochs=2,
            train_batch_size=8,
            val_batch_size=8,
            cache_shard_size=16,
            checkpoint_every_batches=2,
            cache=config["cache"] + "-smoke",
            output=config["output"] + "-smoke",
        )
        if "wandb" in config:
            config["wandb"]["mode"] = "offline"
    stop = {"requested": False}

    def request_stop(signum, frame):
        stop["requested"] = True
        print(f"[SIGNAL {signum}] stopping at next safe batch/image boundary", flush=True)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    manifest, plan = preflight(config)
    if args.stage == "preflight":
        return 0
    with exclusive(config["output"]):
        pin_json(Path(config["output"]) / "requested-config.json", config)
        atomic_write(Path(config["output"]) / "preflight.json", plan)
        if args.stage in ("cache", "all", "smoke"):
            with exclusive(config["cache"]):
                if not prepare_cache(config, stop_requested=lambda: stop["requested"]):
                    print("[STOPPED] completed cache shards preserved; rerun to continue", flush=True)
                    return 130
            gc.collect()
            if config["device"] == "cuda":
                torch.cuda.empty_cache()
        if args.stage in ("train", "all", "smoke"):
            if not run_matrix(config, manifest, args, lambda: stop["requested"]):
                return 130
        if args.stage == "summary":
            experiment = experiment_identity(config, manifest)
            pin_json(Path(config["output"]) / "experiment.json", experiment)
            summarize_matrix(experiment, config["output"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
