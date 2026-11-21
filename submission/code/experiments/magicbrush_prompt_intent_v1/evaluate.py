"""Evaluate all six validation-selected GNN checkpoints on the frozen prompt test split (CPU)."""

from pathlib import Path

import torch

from experiments.magicbrush_prompt_pilot_v1.run import checked_json, example
from tools.edit_model_v1.text import load_store
from tools.emnlp2020_graph_modification_v1.dataset import read, sha256, write
from tools.magicbrush_experiment_v1.evaluation import aggregate_metrics, score_rows
from tools.magicbrush_experiment_v3.model import StructuredPredictor
from tools.magicbrush_experiment_v3.runtime import validate
from tools.magicbrush_experiment_v3.vocabulary import prototype_vectors
from tools.prepare_magicbrush import ROOT, digest

from .dataset import MANIFEST, rows
from .run import OUTPUT, prepare

REPORT = ROOT / "reports/level2/magicbrush-prompt-intent-test-v1"
TEST_CACHE_SHA = "8fb1f532feec844da7c922c964defc21aa1ece1477e656dfb696949deca6631b"


def freeze(arms, validation, config, identity):
    chosen = []
    for seed in config["seeds"]:
        for policy in arms:
            run = OUTPUT / policy / f"seed-{seed}"
            protocol = {**identity, "train_policy": policy}
            expected = {"experiment_sha256": digest(protocol), "model": "gnn", "seed": seed}
            if read(run / "protocol.json") != protocol or read(run / "identity.json") != expected:
                raise ValueError(f"Run identity mismatch: {run}")
            result = read(run / "result.json")
            if result["identity"] != expected or result["test_used"]:
                raise ValueError(f"Unexpected training provenance: {run}")
            for field, filename in [
                ("best_checkpoint_sha256", "best.pt"),
                ("last_checkpoint_sha256", "last.pt"),
                ("predictions_sha256", "predictions-val.json"),
            ]:
                if result[field] != sha256(run / filename):
                    raise ValueError(f"Artifact changed: {run / filename}")
            val = score_rows(validation, read(run / "predictions-val.json"))
            if any(val[k] != v for k, v in result["validation"].items()):
                raise ValueError(f"Saved validation metrics differ: {run}")
            chosen.append(
                {
                    "policy": policy,
                    "seed": seed,
                    "identity": expected,
                    "selected_epoch": result["selected_epoch"],
                    "epochs_completed": result["epochs_completed"],
                    "checkpoint": str((run / "best.pt").relative_to(ROOT)),
                    "sha256": result["best_checkpoint_sha256"],
                    "val_micro_f1": val["micro"]["f1"],
                    "result_sha256": sha256(run / "result.json"),
                }
            )
    write(REPORT / "selection.json", chosen)
    return chosen


def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    arms, validation, config, train_store, identity = prepare()
    chosen = freeze(arms, validation, config, identity)
    cohort = [example(r, r["operations"]) for r in rows("test")]
    cache = Path(config["test_text_cache"])
    metadata = checked_json(cache / "manifest.json", TEST_CACHE_SHA)
    test_store, _ = load_store(cache, metadata["identity"])
    vocabulary = identity["vocabulary"]
    prototypes = prototype_vectors(train_store, vocabulary)
    scores = {policy: [] for policy in arms}
    for entry in chosen:
        best = torch.load(ROOT / entry["checkpoint"], map_location="cpu", weights_only=True)
        if best["identity"] != entry["identity"] or best["epoch"] != entry["selected_epoch"]:
            raise ValueError("Checkpoint differs from validation selection")
        model = StructuredPredictor(kind="gnn", vocabulary=vocabulary, prototypes=prototypes, **config["model"])
        model.load_state_dict(best["model"])
        _, predictions = validate(model, cohort, test_store, config, "cpu")
        metrics = score_rows(cohort, predictions)
        filename = f"{entry['policy']}-seed-{entry['seed']}.json"
        artifact = OUTPUT / "heldout-test-v1" / filename
        write(artifact, predictions)
        scores[entry["policy"]].append(
            {"seed": entry["seed"], "predictions_sha256": sha256(artifact), "metrics": metrics}
        )
    write(
        REPORT / "summary.json",
        {
            "dataset_sha256": sha256(MANIFEST),
            "test_ids": [e.id for e in cohort],
            "device": "cpu",
            "test_cache_manifest_sha256": TEST_CACHE_SHA,
            "evaluator_sha256": sha256(__file__),
            "selection_sha256": sha256(REPORT / "selection.json"),
            "per_seed": scores,
            "aggregate": {k: aggregate_metrics([r["metrics"] for r in entries]) for k, entries in scores.items()},
        },
    )


if __name__ == "__main__":
    main()
