"""Sequential v2 KG pilot/full runs, frozen GQA recipe, explicit W&B batch labels."""

import argparse
import copy
import gc
import json
import signal
import statistics
from pathlib import Path

import torch

from src.datasets.gqa_full import GQAFullDataset, atomic_write, digest, pin_json, verify_cache
from src.datasets.gqa_pilot import select_cohort
from src.knowledge.transe.core import sha256
from src.training import gqa_full
from src.training.gqa_tracking import RunTracker, preserve_rng
from tools import run_gqa_full_ram as ram
from tools import train_gqa_pilot as evaluator
from tools.gqa_wordnet_v2_features import load_v2
from tools.run_gqa_full import exclusive, load_config
from tools.run_gqa_full_parallel import ValidationCache, threaded_collate
from tools.run_gqa_full_prepared import load_prepared_shard, prepare_supervision
from tools.run_gqa_kg_pilot import KINDS, assert_exact, frozen_trainer_extension, summarize, verify_reference

FULL_OUTPUT = Path("runs/level2/gqa-wordnet-full-v2-bs512")


def long_tail_groups(path):
    audit = json.loads(path.read_text())
    counts = {int(row["id"]): int(row["annotation_rows"]["train_full"]) for row in audit["per_predicate"]}
    if set(counts) != set(range(1, 311)):
        raise ValueError("Train frequency report must cover 310 predicates")
    ordered = sorted(counts, key=lambda i: (-counts[i], i))
    return {
        "policy": "descending train annotation count then ID; head 20%, body 40%, tail 40%; fixed before training",
        "source_sha256": sha256(path),
        "counts": {str(i): n for i, n in counts.items()},
        "groups": {"head": ordered[:62], "body": ordered[62:186], "tail": ordered[186:]},
    }


def full_summary(experiment, output, reference_dir):
    results, baseline = [], []
    for seed in experiment["config"]["seeds"]:
        folder = output / "gat-kg" / str(seed)
        result = json.loads((folder / "result.json").read_text())
        identity = {"experiment_sha256": digest(experiment), "model": "gat-kg", "seed": seed}
        if (
            result["identity"] != identity
            or result["status"] != "PASS"
            or sha256(folder / "best.pt") != result["checkpoint_sha256"]
        ):
            raise ValueError("Invalid full KG result/checkpoint")
        old = json.loads((reference_dir / "gat" / str(seed) / "result.json").read_text())
        traces = [[h["sampling_trace_sha256"] for h in r["history"]] for r in (old, result)]
        common = min(map(len, traces))
        if traces[0][:common] != traces[1][:common]:
            raise ValueError("GAT/KG training supervision differs")
        results.append(result)
        baseline.append(old)
    metrics = {}
    for protocol, prefix in (("gc", ""), ("ng", "ng-")):
        for name in ("R@50", "R@100", "mR@50", "mR@100"):
            key = prefix + name
            values = [r["validation"][protocol][key] for r in results]
            old_values = [r["validation"][protocol][key] for r in baseline]
            delta = [a - b for a, b in zip(values, old_values, strict=True)]
            metrics[key] = {
                "values": values,
                "mean": statistics.mean(values),
                "std_sample": statistics.stdev(values),
                "gat_values": old_values,
                "paired_deltas": delta,
                "delta_mean": statistics.mean(delta),
            }
    groups = experiment["long_tail"]["groups"]
    long_tail = {}
    for model, values in (("gat", baseline), ("gat-kg", results)):
        long_tail[model] = {}
        for protocol, prefix in (("gc", ""), ("ng", "ng-")):
            long_tail[model][protocol] = {
                group: {
                    prefix
                    + f"mR@{k}": [
                        statistics.mean(
                            r["validation"][protocol]["per_predicate"][str(i)][prefix + f"R@{k}"] for i in ids
                        )
                        for r in values
                    ]
                    for k in (50, 100)
                }
                for group, ids in groups.items()
            }
    return {
        "status": "PASS",
        "scope": "validation_only_three_seed_main_study",
        "experiment_sha256": digest(experiment),
        "seeds": experiment["config"]["seeds"],
        "metrics": metrics,
        "long_tail": long_tail,
        "training_lengths": [
            {
                "seed": seed,
                "epochs": r["epochs_completed"],
                "selected_epoch": r["selected_epoch"],
                "stopping_reason": r["stopping_reason"],
            }
            for seed, r in zip(experiment["config"]["seeds"], results, strict=True)
        ],
        "matched_GAT_sampling_common_epoch_prefixes": True,
        "test_used": False,
        "interpretation": "Partial-mapping GAT+KG versus existing GAT; no full-run KG-off control or statistical significance claim",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("pilot", "full", "summary"), required=True)
    parser.add_argument("--base-config", type=Path, default=Path("configs/level2/gqa-full-v3-bs512.json"))
    parser.add_argument("--features", type=Path, default=Path("data/knowledge/gqa-wordnet-features-v2"))
    parser.add_argument("--loader-workers", type=int, default=4)
    args = parser.parse_args(argv)
    if not 1 <= args.loader_workers <= 8:
        parser.error("loader-workers must be 1..8")
    pilot = args.stage == "pilot"
    output = Path("runs/level2/gqa-wordnet-pilot-v2") if pilot else FULL_OUTPUT
    report = Path("reports/level2/gqa-wordnet-pilot-v2") if pilot else Path("reports/level2/gqa-wordnet-full-v2")
    base = load_config(args.base_config)
    reference_path = Path(base["output"]) / "experiment.json"
    reference = json.loads(reference_path.read_text())
    verify_reference(reference)
    manifest_path = Path(base["manifest"])
    manifest = json.loads(manifest_path.read_text())
    if sha256(manifest_path) != reference["manifest_sha256"]:
        raise ValueError("Reference dataset manifest changed")
    if sha256(Path(base["cache"]) / "manifest.json") != reference["cache_manifest_sha256"]:
        raise ValueError("Reference cache manifest changed")
    features, valid, feature_manifest = load_v2(args.features, manifest_path)
    config = {
        **copy.deepcopy(base),
        "scope": "pilot" if pilot else "full",
        "output": str(output),
        "models": list(KINDS) if pilot else ["gat-kg"],
        "seeds": [42] if pilot else [42, 123, 3407],
    }
    if pilot:
        config.update(
            epochs=5,
            early_stopping={"enabled": False, "patience": 15, "min_delta": 0.0},
            wandb={**base["wandb"], "mode": "disabled"},
        )
    gqa_full.setup_runtime(config)
    full_metadata = json.loads((Path(base["cache"]) / "manifest.json").read_text())
    cohort = (
        select_cohort(manifest, sha256(manifest_path), 512, 128, 42)
        if pilot
        else {"splits": {k: manifest["splits"][k] for k in ("train", "val")}, "test_used": False}
    )
    extra_sources = [
        Path(__file__),
        Path("tools/gqa_wordnet_v2_features.py"),
        Path("tools/prepare_gqa_wordnet_v2.py"),
        Path("tools/run_gqa_kg_pilot.py"),
        Path("tools/run_gqa_full_prepared.py"),
        Path("tools/run_gqa_full_ram.py"),
        Path("tools/run_gqa_full_parallel.py"),
    ]
    experiment = {
        "schema_version": 2,
        "config": config,
        "dimensions": gqa_full.model_config(manifest),
        "reference_experiment_sha256": sha256(reference_path),
        "cohort": cohort,
        "feature_manifest_sha256": sha256(args.features / "manifest.json"),
        "features_tensor_sha256": feature_manifest["tensor_sha256"],
        "sample_review": feature_manifest["gqa_sample_review"],
        "code_sha256": {
            **reference["code_sha256"],
            **{str(p.relative_to(Path.cwd()) if p.is_absolute() else p): sha256(p) for p in extra_sources},
        },
        "torch_version": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(),
        "protocol": reference["protocol"],
        "sampling": reference["sampling"],
        "long_tail": long_tail_groups(Path("reports/level2/gqa-scale-audit-v1.json")),
        "test_used": False,
        "gqa_context_human_review_completed": False,
        "runtime": {
            "loader_workers": args.loader_workers,
            "collate_threads": 8,
            "numerical_threads": 1,
            "concurrent_model_trainers": 1,
            "validation_cache_max_bytes": int(2.5 * 1024**3),
        },
    }
    if args.stage == "summary":
        pin_json(output / "experiment.json", experiment)
        result = full_summary(experiment, output, Path(base["output"]))
        atomic_write(report / "summary.json", result)
        return 0
    if not pilot:
        pilot_summary = json.loads(Path("reports/level2/gqa-wordnet-pilot-v2/summary.json").read_text())
        if (
            pilot_summary["status"] != "PASS"
            or not pilot_summary["real_data_exact_resume_proof"]
            or pilot_summary["feature_manifest_sha256"] != experiment["feature_manifest_sha256"]
        ):
            raise ValueError("Full training requires a matching successful v2 pilot/resume proof")
    stop = {"requested": False}
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.update(requested=True))
    with exclusive(output), exclusive(Path("runs/level2/.gqa-kg-training-device")):
        pin_json(output / "experiment.json", experiment)
        pin_json(report / "experiment.json", experiment)
        verify_cache(base["cache"], full_metadata["identity"])
        dimensions = experiment["dimensions"]
        datasets = []
        metadata = copy.deepcopy(full_metadata)
        metadata["identity"]["splits"] = cohort["splits"]
        if pilot:
            for split in ("train", "val"):
                dataset = GQAFullDataset(base["cache"], metadata, split)
                try:
                    datasets.append(
                        [prepare_supervision(dataset[i], dimensions["num_classes"]) for i in range(len(dataset))]
                    )
                finally:
                    dataset.close()
        else:
            old_loader = ram.load_shard
            ram.load_shard = load_prepared_shard
            try:
                datasets = [
                    ram.RAMGQADataset(base["cache"], metadata, k, workers=args.loader_workers) for k in ("train", "val")
                ]
            finally:
                ram.load_shard = old_loader
        train, val = datasets
        validation_cache = ValidationCache()

        class BatchLabelTracker(RunTracker):
            def __init__(self, experiment, identity, directory):
                super().__init__(experiment, identity, directory)
                if self.run is not None:
                    with preserve_rng():
                        self.run.tags = tuple(
                            dict.fromkeys(
                                [*self.run.tags, "gqa-wordnet-v2", "batch-512", "partial-human-review", "sequential"]
                            )
                        )
                        self.run.name = f"gqa-wordnet-v2-{identity['model']}-bs512-seed{identity['seed']}"
                        self.run.summary.update({"runtime/concurrent_model_trainers": 1, "runtime/batch_size": 512})

        original_tracker = gqa_full.RunTracker
        results = {}
        try:
            with frozen_trainer_extension(features, valid):
                gqa_full.RunTracker = BatchLabelTracker
                gqa_full.collate, evaluator.collate = threaded_collate, validation_cache
                for kind in config["models"]:
                    for seed in config["seeds"]:
                        if stop["requested"]:
                            return 130
                        result = gqa_full.train_run(
                            kind,
                            seed,
                            train,
                            val,
                            [],
                            experiment,
                            output / kind / str(seed),
                            stop_requested=lambda: stop["requested"],
                        )
                        if result is None:
                            return 130
                        results[kind] = result
                        if pilot:
                            atomic_write(report / f"{kind}.json", result)
                        gc.collect()
                        torch.cuda.empty_cache()
                if pilot:
                    resumed = output / "resume-proof/gat-kg/42"
                    if not (resumed / "last.pt").exists():
                        if (
                            gqa_full.train_run("gat-kg", 42, train, val, [], experiment, resumed, stop_after_batches=1)
                            is not None
                        ):
                            raise AssertionError("Expected interrupted pilot")
                    actual = gqa_full.train_run(
                        "gat-kg", 42, train, val, [], experiment, resumed, stop_requested=lambda: stop["requested"]
                    )
                    if actual is None:
                        return 130
                    for key in ("history", "validation", "initial_state_sha256", "selected_epoch"):
                        assert_exact(actual[key], results["gat-kg"][key])
                    states = [
                        torch.load(p / "last.pt", map_location="cpu", weights_only=True)
                        for p in (resumed, output / "gat-kg/42")
                    ]
                    for key in ("model", "optimizer", "torch_rng", "cuda_rng", "sampler_rng", "best_model"):
                        assert_exact(states[0][key], states[1][key])
                    summary = summarize(results, experiment)
                    summary.update(
                        real_data_exact_resume_proof=True, feature_manifest_sha256=experiment["feature_manifest_sha256"]
                    )
                else:
                    summary = full_summary(experiment, output, Path(base["output"]))
                atomic_write(output / "summary.json", summary)
                atomic_write(report / "summary.json", summary)
        finally:
            gqa_full.RunTracker = original_tracker
            if not pilot:
                for dataset in datasets:
                    dataset.close()
        verify_reference(reference)
        print(f"[PASS] {args.stage} completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
