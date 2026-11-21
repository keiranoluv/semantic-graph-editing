"""Sequential real-data GAT / KG-off / WordNet-KG pilot using the frozen trainer.

All extensions live outside src/: adding even a src file would change the
historical experiment's glob-based identity. Runtime substitutions are scoped
and recorded, and never change an existing run's identity or source files.
"""

import argparse
import copy
import gc
import json
import signal
from contextlib import contextmanager
from pathlib import Path

import torch

from src.datasets.gqa_full import GQAFullDataset, atomic_write, digest, pin_json, verify_cache
from src.datasets.gqa_pilot import select_cohort
from src.knowledge.transe.core import sha256
from src.models.gat_kg import KnowledgeGAT
from src.training import gqa_full
from tools import train_gqa_pilot as evaluator
from tools.export_gqa_wordnet import load_features
from tools.run_gqa_full import exclusive, load_config
from tools.run_gqa_full_prepared import prepare_supervision, prepared_collate

KINDS = ("gat", "gat-kg-off", "gat-kg")


def create_model(kind, dimensions, features, valid):
    if kind == "gat":
        return evaluator.build_model(kind, dimensions)
    if kind not in KINDS:
        raise ValueError(f"Unsupported GQA KG model: {kind}")
    settings = {
        **dimensions,
        "kg_dim": features.shape[1],
        "use_knowledge": kind == "gat-kg",
        "message_edges": "full",
        "num_heads": 4,
        "num_layers": 2,
        "dropout": 0.1,
    }
    return KnowledgeGAT(settings, features, valid)


@contextmanager
def frozen_trainer_extension(features, valid):
    """Only model creation/dispatch and equivalent prepared collation change."""
    original = (gqa_full.create_model, gqa_full.forward, gqa_full.evaluate, gqa_full.collate, evaluator.collate)
    original_forward, original_evaluate = gqa_full.forward, gqa_full.evaluate
    gqa_full.create_model = lambda kind, dimensions, vectors: create_model(kind, dimensions, features, valid)
    gqa_full.forward = lambda model, kind, batch, device: original_forward(model, "gat", batch, device)
    gqa_full.evaluate = lambda model, kind, *a, **kw: original_evaluate(model, "gat", *a, **kw)
    gqa_full.collate = evaluator.collate = prepared_collate
    try:
        yield
    finally:
        (gqa_full.create_model, gqa_full.forward, gqa_full.evaluate, gqa_full.collate, evaluator.collate) = original


def verify_reference(reference):
    """Ensure the completed 15-run implementation still has its pinned hashes."""
    for name, expected in reference["code_sha256"].items():
        if sha256(Path(name)) != expected:
            raise ValueError(f"Frozen reference source changed: {name}")


def assert_exact(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    elif isinstance(a, dict):
        if a.keys() != b.keys():
            raise AssertionError("State keys differ")
        for k in a:
            assert_exact(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        if len(a) != len(b):
            raise AssertionError("State lengths differ")
        for x, y in zip(a, b, strict=True):
            assert_exact(x, y)
    elif a != b:
        raise AssertionError("State values differ")


def summarize(results, experiment):
    traces = [[row["sampling_trace_sha256"] for row in r["history"]] for r in results.values()]
    if any(t != traces[0] for t in traces):
        raise AssertionError("Pilot controls did not receive identical supervision")
    if results["gat-kg"]["initial_state_sha256"] != results["gat-kg-off"]["initial_state_sha256"]:
        raise AssertionError("KG on/off initialization differs")
    return {
        "status": "PASS",
        "scope": "pilot_not_main_experiment",
        "experiment_sha256": digest(experiment),
        "matched_sampling": True,
        "matched_kg_on_off_initialization": True,
        "models": {
            kind: {
                "selected_epoch": r["selected_epoch"],
                "validation": r["validation"],
                "parameters": r["parameters"],
                "checkpoint_sha256": r["checkpoint_sha256"],
                "peak_cuda_allocated_bytes": r["peak_cuda_allocated_bytes_this_invocation"],
            }
            for kind, r in results.items()
        },
        "interpretation": "One seed, 512 train / 128 val, 5 epochs; pipeline check only, not evidence of KG benefit",
        "test_used": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=Path("configs/level2/gqa-full-v3-bs512.json"))
    parser.add_argument("--features", type=Path, default=Path("data/knowledge/gqa-wordnet-features-draft-v1"))
    parser.add_argument("--output", type=Path, default=Path("runs/level2/gqa-wordnet-pilot-v1"))
    parser.add_argument("--report", type=Path, default=Path("reports/level2/gqa-wordnet-pilot-v1"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    base = load_config(args.base_config)
    reference_path = Path(base["output"]) / "experiment.json"
    reference = json.loads(reference_path.read_text())
    verify_reference(reference)
    manifest_path = Path(base["manifest"])
    if sha256(manifest_path) != reference["manifest_sha256"]:
        raise ValueError("GQA manifest differs from reference")
    cache_manifest_path = Path(base["cache"]) / "manifest.json"
    if sha256(cache_manifest_path) != reference["cache_manifest_sha256"]:
        raise ValueError("GQA cache differs from reference")
    features, valid, feature_manifest = load_features(args.features, manifest_path)
    config = {
        **copy.deepcopy(base),
        "scope": "pilot",
        "output": str(args.output),
        "models": list(KINDS),
        "seeds": [42],
        "epochs": 5,
        "device": args.device,
        "early_stopping": {"enabled": False, "patience": 15, "min_delta": 0.0},
        "wandb": {**base["wandb"], "mode": "disabled"},
    }
    gqa_full.setup_runtime(config)
    manifest = json.loads(manifest_path.read_text())
    cohort = select_cohort(manifest, sha256(manifest_path), 512, 128, 42)
    stop = {"requested": False}
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.update(requested=True))
    with exclusive(args.output), exclusive(Path("runs/level2/.gqa-kg-training-device")):
        pin_json(args.output / "requested-config.json", config)
        full_metadata = json.loads(cache_manifest_path.read_text())
        verify_cache(base["cache"], full_metadata["identity"])
        dimensions = gqa_full.model_config(manifest)
        sources = [Path(name) for name in reference["code_sha256"]] + [
            Path(name)
            for name in (
                "tools/run_gqa_kg_pilot.py",
                "tools/export_gqa_wordnet.py",
                "tools/prepare_gqa_wordnet.py",
                "tools/run_gqa_full_prepared.py",
            )
        ]
        experiment = {
            "schema_version": 1,
            "config": config,
            "dimensions": dimensions,
            "reference_experiment_sha256": sha256(reference_path),
            "cohort": cohort,
            "feature_manifest_sha256": sha256(args.features / "manifest.json"),
            "features_tensor_sha256": feature_manifest["tensor_sha256"],
            "code_sha256": {str(p): sha256(p) for p in sources},
            "torch_version": str(torch.__version__),
            "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
            "protocol": reference["protocol"],
            "sampling": reference["sampling"],
            "test_used": False,
            "gqa_context_human_review_completed": False,
        }
        pin_json(args.output / "experiment.json", experiment)
        pin_json(args.report / "experiment.json", experiment)
        metadata = copy.deepcopy(full_metadata)
        metadata["identity"]["splits"] = cohort["splits"]
        graphs = []
        for split in ("train", "val"):
            dataset = GQAFullDataset(base["cache"], metadata, split)
            try:
                graphs.append([prepare_supervision(dataset[i], dimensions["num_classes"]) for i in range(len(dataset))])
            finally:
                dataset.close()
        train, val = graphs
        coverage = {}
        for split, items in zip(("train", "val"), graphs, strict=True):
            labels = torch.cat([g.object_labels for g in items])
            coverage[split] = {
                "objects": len(labels),
                "objects_with_valid_KG": int(valid[labels].sum()),
                "object_KG_coverage": float(valid[labels].float().mean()),
            }
        atomic_write(args.report / "node-coverage.json", coverage)
        results = {}
        with frozen_trainer_extension(features, valid):
            for kind in KINDS:
                result = gqa_full.train_run(
                    kind,
                    42,
                    train,
                    val,
                    [],
                    experiment,
                    args.output / kind / "42",
                    stop_requested=lambda: stop["requested"],
                )
                if result is None:
                    return 130
                results[kind] = result
                atomic_write(args.report / f"{kind}.json", result)
                gc.collect()
                if args.device == "cuda":
                    torch.cuda.empty_cache()
            resumed = args.output / "resume-proof" / "gat-kg" / "42"
            if not (resumed / "last.pt").exists():
                if (
                    gqa_full.train_run("gat-kg", 42, train, val, [], experiment, resumed, stop_after_batches=1)
                    is not None
                ):
                    raise AssertionError("Expected a diagnostic interrupt")
            actual = gqa_full.train_run(
                "gat-kg", 42, train, val, [], experiment, resumed, stop_requested=lambda: stop["requested"]
            )
            if actual is None:
                return 130
            for key in ("history", "validation", "initial_state_sha256", "selected_epoch"):
                assert_exact(actual[key], results["gat-kg"][key])
            left, right = [
                torch.load(p / "last.pt", map_location="cpu", weights_only=True)
                for p in (resumed, args.output / "gat-kg" / "42")
            ]
            for key in ("model", "optimizer", "torch_rng", "cuda_rng", "sampler_rng", "best_model"):
                assert_exact(left[key], right[key])
        summary = summarize(results, experiment)
        summary["real_data_exact_resume_proof"] = True
        summary["node_coverage"] = coverage
        atomic_write(args.output / "summary.json", summary)
        atomic_write(args.report / "summary.json", summary)
        verify_reference(reference)
        print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
