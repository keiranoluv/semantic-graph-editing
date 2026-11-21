"""Aggregate sealed EMNLP Task B results without redistributing dataset records."""

import argparse
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from common.editing.operations import KINDS
from task2_graph_editing.he2020.preprocessing.data import source_fingerprint
from common.io import digest, read, sha256, write



def distribution(values):
    values = [v for v in values if v is not None]
    return {
        "mean": statistics.mean(values) if values else None,
        "sample_std": statistics.stdev(values) if len(values) > 1 else None,
        "values": values,
    }


def aggregate(metrics):
    output = {
        key: distribution([m[key] for m in metrics])
        for key in (
            "exact_operation_set_accuracy",
            "exact_target_graph_accuracy",
            "valid_transaction_rate",
            "simple_exact_operation_set_accuracy",
            "macro_f1_supported_types",
        )
    }
    output["micro"] = {k: distribution([m["micro"][k] for m in metrics]) for k in ("precision", "recall", "f1")}
    output["per_type"] = {
        kind: {
            k: distribution([m["per_type"][kind][k] for m in metrics])
            for k in ("precision", "recall", "f1", "tp", "fp", "fn", "support")
        }
        for kind in KINDS
    }
    return output


def dataset_audit(bundle):
    per_split = {}
    pairs = defaultdict(set)
    for row in bundle["records"]:
        if row.get("raw_source") is not None and row.get("raw_target") is not None:
            pair = digest([source_fingerprint(row["raw_source"]), " ".join(row["instruction"].split()).casefold()])
            pairs[pair].add(source_fingerprint(row["raw_target"]))
    for split in sorted({r["split"] for r in bundle["records"]}):
        rows = [r for r in bundle["records"] if r["split"] == split]
        usable = [r for r in rows if not r["exclusion"]]
        per_split[split] = {
            "unique_source_groups_all": len({r.get("source_group_id") for r in rows}),
            "unique_source_groups_usable": len({r.get("source_group_id") for r in usable}),
            "raw_source_attribute_pairs_all": sum(
                any(e["data"]["feature"] == "att" for e in r["raw_source"]["edges"]) for r in rows if r["raw_source"]
            ),
            "raw_source_self_loops_all": sum(
                e["u"] == e["v"] for r in rows if r["raw_source"] for e in r["raw_source"]["edges"]
            ),
            "usable_operation_count_histogram": dict(Counter(len(r["operations"]) for r in usable)),
            "usable_alignment_scores": dict(Counter(str(r["alignment"]["score"]) for r in usable)),
        }
    return {
        "splits": per_split,
        "same_source_instruction_with_nonisomorphic_targets": sum(len(targets) > 1 for targets in pairs.values()),
        "conflict_check_scope": "within_this_bundle_only_audit_targets_never_used_to_group_or_repair",
    }


def classify(case):
    if case["correct"]:
        return "correct"
    if case["error"]:
        return case["error"].split(":")[0]
    expected, predicted = case["expected"], case["predicted"]
    if Counter(o["kind"] for o in expected) != Counter(o["kind"] for o in predicted):
        return "wrong_operation_inventory"

    def entity(o):
        return (o["before"] or o["after"])["id"]

    if Counter((o["kind"], entity(o)) for o in expected) != Counter((o["kind"], entity(o)) for o in predicted):
        return "wrong_entity_identity"
    return "wrong_payload"


def sanitized_cases(cases):
    return [
        {
            "id": c["id"],
            "correct": c["correct"],
            "category": classify(c),
            "reference_OOV": c["reference_OOV"],
            "expected_kinds": dict(Counter(o["kind"] for o in c["expected"])),
            "predicted_kinds": dict(Counter(o["kind"] for o in c["predicted"])),
        }
        for c in cases
    ]


def report(config, output):
    directory, dataset, output = Path(config["output"]), Path(config["dataset"]), Path(output)
    freeze = read(directory / "frozen-checkpoints.json")
    experiment = read(directory / "experiment.json")
    test = read(directory / "test-summary.json")
    results, cases = {}, {}
    for model in config["models"]:
        train_results = [read(directory / f"{model}-seed{seed}" / "result.json") for seed in config["seeds"]]
        test_results = [read(directory / f"{model}-seed{seed}" / "test-result.json") for seed in config["seeds"]]
        results[model] = {
            "validation": aggregate([r["validation"] for r in train_results]),
            "test": aggregate([r["metrics"] for r in test_results]),
            "seeds": config["seeds"],
            "best_epochs": [r["best_epoch"] for r in train_results],
            "runtime_train_and_val_seconds": [r["runtime_seconds"] for r in train_results],
            "validation_ablations": {
                name: aggregate([r["validation_ablations"][name] for r in train_results])
                for name in train_results[0]["validation_ablations"]
            },
            "test_vocabulary_OOV": test_results[0]["metrics"]["reference_programs_with_train_vocabulary_OOV"],
            "test_closed_vocabulary_exact_match_upper_bound": test_results[0]["metrics"][
                "closed_vocabulary_exact_match_upper_bound"
            ],
        }
        # Seed 42 was predeclared, not selected by test performance.
        primary = directory / f"{model}-seed42"
        predictions = read(primary / "test-predictions.json")
        cases[model] = {
            "seed": 42,
            "test_categories": dict(Counter(classify(c) for c in predictions)),
            "selected_five_correct_five_wrong": sanitized_cases(read(primary / "test-cases.json")),
        }
    for name in ("trainval", "test"):
        write(output / f"{name}-summary.json", read(dataset / f"{name}-summary.json"))
        write(output / f"{name}-audit.json", dataset_audit(read(dataset / f"{name}.json")))
    provenance = {
        "experiment": experiment,
        "frozen_checkpoints": freeze,
        "data_receipts": {name: read(dataset / f"{name}-receipt.json") for name in ("trainval", "test")},
        "features": {
            name: {k: v for k, v in read(dataset / f"clip-{name}/manifest.json").items() if k != "text_keys"}
            for name in ("trainval", "test")
        },
        "report_code_sha256": sha256(Path(__file__)),
        "test_summary_sha256": sha256(directory / "test-summary.json"),
    }
    # Train vocabulary is a converted dataset artifact; retain only its hash/count.
    vocabulary = provenance["experiment"].pop("vocabulary")
    provenance["experiment"]["vocabulary_size_including_specials"] = len(vocabulary) + 4
    provenance["experiment"]["vocabulary_sha256"] = digest(vocabulary)
    write(output / "provenance.json", provenance)
    write(
        output / "metrics.json",
        {
            "models": results,
            "baselines": {k: v for k, v in test.items() if k.endswith("baseline")},
            "test_opened_after_freeze": test["test_opened_after_freeze"],
            "aggregation": "mean_and_sample_std_across_three_seeds_same_examples",
        },
    )
    write(output / "error-analysis.json", cases)
    print(f"Wrote aggregate statistics to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/level2/emnlp2020-task-b-v2.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/level2/emnlp2020-task-b-v2"))
    args = parser.parse_args()
    report(read(args.config), args.output)
