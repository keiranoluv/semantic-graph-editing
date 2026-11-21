"""Isolated reuse of the frozen trainer, VG150 evaluator and explicit one-run CLI."""

import argparse
import copy
import importlib.metadata
import importlib.util
import json
import signal
import uuid
from pathlib import Path

import torch

from src.datasets.gqa_full import atomic_write, pin_json
from src.evaluation.reference_predcls import evaluate_reference_image, summarize_reference
from src.knowledge.export import load_class_features
from src.knowledge.transe.core import sha256
from src.models.gat_kg import KnowledgeGAT
from src.training import gqa_full
from src.training.gqa_pilot import cross_entropy_sum, forward
from src.training.gqa_tracking import RunTracker, preserve_rng
from tools.prepare_magicbrush import ROOT, digest, encode, fixed
from tools.run_gqa_full import exclusive
from tools.run_gqa_full_prepared import prepared_collate
from tools.vg150_penet_v1.data import cohort, datasets, read_splits
from tools.vg150_penet_v1.prepare import MODELS, load_prepared, prepare, validate

DEFAULT_CONFIG = ROOT / "configs/level2/vg150-penet-v1.json"


def create_model(kind, dimensions, vectors, features, valid):
    if kind == "gat-kg":
        return KnowledgeGAT(
            {
                **dimensions,
                "kg_dim": features.shape[1],
                "use_knowledge": True,
                "message_edges": "full",
                "num_heads": 4,
                "num_layers": 2,
                "dropout": 0.1,
            },
            features,
            valid,
        )
    if kind not in MODELS:
        raise ValueError("Unknown VG150 model")
    return gqa_full.create_model(kind, dimensions, vectors)


@torch.inference_mode()
def evaluate(model, kind, graphs, config, device, *, include_ng=False):
    model.eval()
    results = {"gc": [], "ng": []}
    loss_sum, count = 0.0, 0
    for begin in range(0, len(graphs), config["batch_size"]):
        group = graphs[begin : begin + config["batch_size"]]
        batch, slices = prepared_collate(group, config["num_classes"])
        logits = forward(model, "gat" if kind == "gat-kg" else kind, batch, device).cpu()
        loss_sum += float(cross_entropy_sum(logits, batch["targets"]))
        count += len(logits)
        for graph, (start, end) in zip(group, slices, strict=True):
            for protocol in ("gc", "ng") if include_ng else ("gc",):
                results[protocol].append(
                    evaluate_reference_image(
                        logits[start:end],
                        graph.edge_label_index,
                        graph.gt_relation_index,
                        graph.gt_predicate,
                        graph.boxes,
                        graph.object_labels,
                        graph_constraint=protocol == "gc",
                    )
                )
    if not count:
        raise ValueError("No validation candidate pairs")
    output = {"loss_all_pairs": loss_sum / count}
    for protocol in ("gc", "ng") if include_ng else ("gc",):
        summary = summarize_reference(
            results[protocol], num_predicates=config["num_classes"] - 1, graph_constraint=protocol == "gc"
        )
        summary[
            "protocol"
        ] = f"VG150_PredCls_{protocol.upper()}_reference_iou50_duplicate_GT_rows_image_macro_fixed50_v1"
        prefix = "" if protocol == "gc" else "ng-"
        if "tail_ids" in config:
            summary["tail40_mR@50"] = sum(
                summary["per_predicate"][str(i)][prefix + "R@50"] for i in config["tail_ids"]
            ) / len(config["tail_ids"])
        output[protocol] = summary
    return output


class Tracker(RunTracker):
    """Reuse durable-epoch/RNG-safe logging, with VG150 names and grouping."""

    def __init__(self, experiment, identity, directory):
        self.directory, self.run, self.latest_epoch = Path(directory), None, 0
        self.settings = experiment["config"]["wandb"]
        self.mode = self.settings["mode"]
        if self.mode == "disabled":
            return
        with preserve_rng():
            import wandb

            stable = digest({"identity": identity, "directory": str(self.directory.resolve())})[:24]
            tracking = {
                "logical_id": stable,
                "identity": identity,
                "mode": self.mode,
                "entity": self.settings["entity"],
                "project": self.settings["project"],
            }
            pin_json(self.directory / "wandb-identity.json", tracking)
            physical = stable if self.mode == "online" else f"{stable}-{uuid.uuid4().hex[:8]}"
            local = self.directory / "tracking"
            local.mkdir(exist_ok=True)
            batch = experiment["config"]["train_batch_size"]
            scope = experiment["config"]["scope"]
            self.run = wandb.init(
                entity=self.settings["entity"],
                project=self.settings["project"],
                id=physical,
                resume="allow" if self.mode == "online" else None,
                mode=self.mode,
                dir=str(local),
                name=f"vg150-penet-v1-{scope}-{identity['model']}-bs{batch}-seed{identity['seed']}",
                group=f"vg150-penet-v1-{scope}-{identity['experiment_sha256'][:12]}",
                job_type="training",
                config={"experiment": experiment, "run_identity": identity, "metric_units": "fractions"},
                tags=[
                    "vg150",
                    "penet-adapted",
                    "multilabel",
                    scope,
                    identity["model"],
                    f"batch-{batch}",
                    f"seed-{identity['seed']}",
                ],
                save_code=False,
                settings=wandb.Settings(init_timeout=30, disable_git=True, console="off"),
            )
            self.run.define_metric("epoch")
            for key in ("train/*", "val/*", "best/*", "early_stopping/*"):
                self.run.define_metric(key, step_metric="epoch")
            self.run.summary.update({"logical_id": stable, "execution_status": "running"})
            self.info = {
                **tracking,
                "run_id": physical,
                "url": self.run.url if self.mode == "online" else None,
                "files_dir": self.run.dir,
                "execution_status": "running",
            }
            atomic_write(self.directory / "wandb-latest.json", self.info)


def isolated_trainer(features, valid):
    """Private module namespace: never mutate frozen trainer globals or source."""
    spec = importlib.util.spec_from_file_location("_vg150_penet_v1_frozen_trainer", Path(gqa_full.__file__))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.create_model = lambda kind, dimensions, vectors: create_model(kind, dimensions, vectors, features, valid)
    module.forward = lambda model, kind, batch, device: forward(
        model, "gat" if kind == "gat-kg" else kind, batch, device
    )
    module.evaluate = evaluate
    module.collate = prepared_collate
    module.RunTracker = Tracker
    return module


def source_hashes(config):
    reference = json.loads(Path(config["reference_experiment"]).read_text())
    for name, expected in reference["code_sha256"].items():
        if sha256(Path(name)) != expected:
            raise ValueError(f"Historical GQA source changed: {name}")
    extra = [
        *Path(__file__).parent.glob("*.py"),
        ROOT / "tools/run_vg150_penet.py",
        ROOT / "tools/run_gqa_full_prepared.py",
        ROOT / "tools/run_gqa_full_ram.py",
    ]
    return {
        **reference["code_sha256"],
        **{str(p.relative_to(ROOT)) if p.is_absolute() else str(p): sha256(p) for p in extra},
    }


def experiment(config, metadata, *, pilot=False):
    cfg = copy.deepcopy(config)
    cfg["scope"] = "pilot" if pilot else "full"
    if pilot:
        cfg.update(
            epochs=config["pilot"]["epochs"],
            seeds=[42],
            early_stopping={"enabled": False, "patience": 15, "min_delta": 0.0},
        )
    selected = cohort(read_splits(config), pilot=pilot)
    dimensions = gqa_full.model_config({"vocabulary": metadata["vocabulary"]})
    support = metadata["train_statistics"]["predicate_support"]
    head = sorted(range(1, 51), key=lambda i: (-support[str(i)], i))[:10]
    dimensions["tail_ids"] = [i for i in range(1, 51) if i not in head]
    return {
        "schema_version": 1,
        "dataset": "VG150",
        "config": cfg,
        "dimensions": dimensions,
        "base_config_sha256": digest(config),
        "cohort": selected,
        "prepared_manifest_sha256": sha256(Path(config["prepared"]) / "manifest.json"),
        "artifacts": metadata["artifacts"],
        "glove_arrays_sha256": metadata["glove"]["arrays_sha256"],
        "code_sha256": source_hashes(config),
        "versions": {k: importlib.metadata.version(k) for k in ("torch", "torch-geometric", "numpy", "h5py")},
        "sampling": "distinct positive predicates uniform soft CE; all positive pairs + up to 3x negatives; empty positives up to 32 negatives",
        "topology": "all ordered non-self object pairs, independent of GT relations",
        "protocol": "unchanged reference PredCls GC/NG iou50; all duplicate GT rows; fixed 50-class image-macro mR",
        "selection": "validation GC mR@50; ties keep earlier epoch; patience 15/min_delta0",
        "long_tail": {"head_ids_train_top10": head, "tail_ids": dimensions["tail_ids"]},
        "interpretation": "PE-Net adapted frozen-CLIP pair-feature proxy; not original detector/union ROI reproduction",
        "GloVe_is_external_prior_not_KG": True,
        "test_used": False,
    }


def study_path(config, pilot=False):
    return Path(config["output"]) / ("pilot" if pilot else "full")


def require_pilots(config, metadata):
    expected = experiment(config, metadata, pilot=True)
    path = study_path(config, True)
    for kind in MODELS:
        run = path / kind / "42"
        if not (run / "result.json").is_file():
            raise ValueError(f"Main training needs completed matched pilot: {kind}/42")
        result = json.loads((run / "result.json").read_text())
        if (
            result["status"] != "PASS"
            or result["identity"] != {"experiment_sha256": digest(expected), "model": kind, "seed": 42}
            or sha256(run / "best.pt") != result["checkpoint_sha256"]
        ):
            raise ValueError("Pilot identity/checkpoint mismatch")
    return summary(config, metadata, pilot=True)


def summary(config, metadata, *, pilot=False):
    study = experiment(config, metadata, pilot=pilot)
    result = gqa_full.summarize_matrix(study, study_path(config, pilot))
    result.update(
        dataset="VG150",
        interpretation="validation only; new matched multi-positive supervision",
        protocol=study["protocol"],
    )
    tails = {}
    for kind in MODELS:
        values = []
        for seed in study["config"]["seeds"]:
            row = json.loads((study_path(config, pilot) / kind / str(seed) / "result.json").read_text())
            values.append(row["validation"]["gc"]["tail40_mR@50"])
        import numpy as np

        tails[kind] = {
            "values": values,
            "mean": float(np.mean(values)),
            "std_sample": float(np.std(values, ddof=1)) if len(values) > 1 else None,
        }
    result["tail40_mR@50"] = tails
    atomic_write(study_path(config, pilot) / "summary.json", result)
    fixed(Path(config["report"]) / ("pilot-summary.json" if pilot else "full-summary.json"), encode(result))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=("prepare", "plan", "pilot", "train", "summary"), default="plan")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--seed", type=int, choices=(42, 123, 3407))
    parser.add_argument("--scope", choices=("pilot", "full"), default="full", help="plan/summary scope only")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    validate(config)
    if args.stage == "prepare":
        with exclusive(Path(config["prepared"])):
            prepare(config)
        return 0
    ready = (Path(config["prepared"]) / "manifest.json").is_file()
    if not ready:
        print(json.dumps({"ready": False, "reason": "Run CPU --stage prepare first", "GPU_started": False}))
        return 2
    metadata, arrays = load_prepared(config)
    if args.stage == "plan":
        study = experiment(config, metadata, pilot=args.scope == "pilot")
        completed = [kind for kind in MODELS if (study_path(config, True) / kind / "42/result.json").exists()]
        print(
            json.dumps(
                {
                    "stage": "CPU_plan_no_GPU",
                    "ready_for_pilot": True,
                    "pilots_with_result_files": completed,
                    "full_requires_all_six_verified_pilots": True,
                    "scope": args.scope,
                    "models": list(MODELS),
                    "seeds": study["config"]["seeds"],
                    "images": {k: len(v) for k, v in study["cohort"].items()},
                    "batch_size": config["train_batch_size"],
                    "output": str(study_path(config, args.scope == "pilot")),
                    "GPU_started": False,
                    "test_used": False,
                },
                indent=2,
            )
        )
        return 0
    if args.stage == "summary":
        summary(config, metadata, pilot=args.scope == "pilot")
        return 0
    if args.model is None or args.seed is None:
        parser.error("pilot/train requires explicit --model and --seed; one run only")
    pilot = args.stage == "pilot"
    if pilot and args.seed != 42:
        parser.error("Pilot uses only seed 42")
    study = experiment(config, metadata, pilot=pilot)
    stop = {"requested": False}
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.update(requested=True))
    with exclusive(ROOT / "runs/level2/.gqa-kg-training-device"), exclusive(study_path(config, pilot)):
        if not pilot:
            require_pilots(config, metadata)
        gqa_full.setup_runtime(study["config"])
        pin_json(study_path(config, pilot) / "experiment.json", study)
        features, valid, _ = load_class_features(Path(config["kg_features"]))
        items = datasets(config, study["cohort"])
        try:
            trainer = isolated_trainer(features, valid)
            result = trainer.train_run(
                args.model,
                args.seed,
                items["train"],
                items["val"],
                [torch.from_numpy(a) for a in arrays],
                study,
                study_path(config, pilot) / args.model / str(args.seed),
                stop_requested=lambda: stop["requested"],
            )
            return 0 if result is not None else 130
        finally:
            for item in items.values():
                item.close()
