"""CPU-only diagnosis of frozen v3 validation predictions; never loads test labels."""
import argparse
import random
from collections import Counter
from pathlib import Path

from tools.edit_model_v1.codec import canonical_actions, decode_actions
from tools.emnlp2020_graph_modification_v1.dataset import read, sha256, write
from tools.magicbrush_experiment_v1.evaluation import score_rows, verify_receipt
from tools.magicbrush_experiment_v3.pipeline import DEFAULT_CONFIG, preflight
from tools.magicbrush_experiment_v3.vocabulary import coverage
from tools.prepare_magicbrush import ROOT, digest

OUTPUT = ROOT / "reports/level2/magicbrush-manual-v3-error-analysis-v1"
SAMPLE_SEED = 20261114
EXISTING = {"delete_node", "replace_node", "delete_edge", "replace_edge"}


def predictions(example, row):
    ops = decode_actions(row["tokens"], example.source)
    return canonical_actions(example.source, ops)


def diagnose(example, predicted):
    by_entity = {(o.kind.split("_")[1], o.entity_id): o for o in predicted if o.kind in EXISTING}
    gt_entities = {(o.kind.split("_")[1], o.entity_id) for o in example.operations if o.kind in EXISTING}
    counts, details = Counter(), []
    for expected in example.operations:
        if expected.kind not in EXISTING:
            continue
        counts["gt_existing_operations"] += 1
        p = by_entity.get((expected.kind.split("_")[1], expected.entity_id))
        counts["existing_entity_touched"] += p is not None
        same = p is not None and p.kind == expected.kind
        counts["existing_kind_and_id_correct"] += same
        counts["existing_exact_payload_correct"] += p == expected
        status = (
            "entity_not_edited"
            if p is None
            else "wrong_operation_kind"
            if not same
            else "exact"
            if p == expected
            else "wrong_payload"
        )
        counts[status] += 1
        field_errors = []
        if same and expected.after is not None:
            fields = ("label", "attributes") if expected.kind.endswith("node") else ("subject", "predicate", "object")
            for field in fields:
                counts[f"matched_{expected.kind}_fields"] += 1
                if getattr(p.after, field) != getattr(expected.after, field):
                    field_errors.append(field)
                    counts[f"wrong_{expected.kind}_{field}"] += 1
            if expected.kind == "replace_node":
                counts["replace_node_correct_id_and_kind"] += 1
                counts["replace_node_correct_label_given_id_kind"] += p.after.label == expected.after.label
                counts["replace_node_correct_attributes_given_id_kind"] += (
                    p.after.attributes == expected.after.attributes
                )
        details.append(
            {"kind": expected.kind, "entity_id": expected.entity_id, "status": status, "wrong_fields": field_errors}
        )
    counts["predicted_existing_ops_on_unrequested_entities"] = sum(key not in gt_entities for key in by_entity)
    gt_added = Counter(o.after.label for o in example.operations if o.kind == "add_node")
    p_added = Counter(o.after.label for o in predicted if o.kind == "add_node")
    counts["gt_added_nodes"] = sum(gt_added.values())
    counts["predicted_added_nodes"] = sum(p_added.values())
    counts["added_node_label_overlap_ignoring_ids_and_attributes"] = sum((gt_added & p_added).values())
    counts["exact_operation_tp"] = sum((Counter(example.operations) & Counter(predicted)).values())
    counts["gt_operations"] = len(example.operations)
    counts["predicted_operations"] = len(predicted)
    counts["predicted_self_loop_edges"] = sum(
        o.after is not None and o.kind.endswith("edge") and o.after.subject == o.after.object for o in predicted
    )
    return dict(counts), details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    config, examples, preflight_report, identity, paths = preflight(DEFAULT_CONFIG)
    output = args.output
    run = ROOT / config["output"]
    exp = digest(read(run / "experiment.json"))
    # Sample accepted validation pairs before examining performance; no test cases.
    selected = sorted(
        random.Random(SAMPLE_SEED).sample(sorted(examples["val"], key=lambda e: e.id), 20), key=lambda e: e.id
    )
    selected_ids = {e.id for e in selected}
    manifest = read(paths["trainval_manifest"])
    metadata = {r["id"]: r for r in manifest["records"]}
    report = {
        "scope": "validation only; saved outputs and image/annotation inspection; no model inference or GPU",
        "experiment_sha256": exp,
        "sample_seed": SAMPLE_SEED,
        "sampling": "random.sample of 20 out of 67 accepted validation pairs sorted by ID; pair sampling, not group-balanced",
        "sample_ids": [e.id for e in selected],
        "sample_groups": len({e.group for e in selected}),
        "primary_qualitative_model": "gnn",
        "primary_qualitative_seed": 42,
        "primary_selection_reason": "fixed first seed, not selected using test performance",
        "vocabulary_coverage": {s: coverage(es, identity["vocabulary"]) for s, es in examples.items()},
        "models": {},
        "input_hashes": {},
        "diagnostic_warning": "Kind/ID and added-label counts are supplemental diagnostics, not replacement grading metrics. Added-node label overlap does not establish instance alignment.",
    }
    packets = {}
    for kind in config["models"]:
        report["models"][kind] = {}
        for seed in config["seeds"]:
            directory = run / kind / str(seed)
            verify_receipt(directory, exp, kind, seed)
            rows = read(directory / "predictions-val.json")
            metrics = score_rows(examples["val"], rows)
            result = read(directory / "result.json")
            assert all(metrics[k] == v for k, v in result["validation"].items())
            total, sample = Counter(), Counter()
            per_case = []
            for e, row in zip(examples["val"], rows, strict=True):
                predicted = predictions(e, row)
                counts, details = diagnose(e, predicted)
                total.update(counts)
                if e.id in selected_ids:
                    sample.update(counts)
                    if kind == "gnn" and seed == 42:
                        meta = metadata[e.id]
                        packets[e.id] = {
                            "id": e.id,
                            "instruction": e.instruction,
                            "group": e.group,
                            "source_image": meta["source_image"],
                            "target_image": meta["target_image"],
                            "source_graph": e.source.to_dict(),
                            "target_graph": e.target.to_dict(),
                            "expected_operations": [o.to_dict() for o in e.operations],
                            "predicted_operations": [o.to_dict() for o in predicted],
                            "raw_tokens": row["raw_tokens"],
                            "constraints": row["constraints"],
                            "counts": counts,
                            "existing_operation_diagnosis": details,
                        }
                per_case.append({"id": e.id, "counts": counts, "existing_operation_diagnosis": details})
            # Cross-check the new diagnostic against the frozen official scorer.
            assert total["exact_operation_tp"] == metrics["micro"]["tp"]
            assert total["gt_operations"] == metrics["micro"]["support"]
            assert total["gt_existing_operations"] == (
                total["existing_kind_and_id_correct"] + total["entity_not_edited"] + total["wrong_operation_kind"]
            )
            assert total["existing_kind_and_id_correct"] == total["exact"] + total["wrong_payload"]
            report["models"][kind][str(seed)] = {
                "official_validation": metrics,
                "all_67": dict(total),
                "sample_20": dict(sample),
                "per_case": per_case,
            }
            report["input_hashes"][str(directory.relative_to(ROOT) / "predictions-val.json")] = sha256(
                directory / "predictions-val.json"
            )
    frequencies = {"added_attributes": Counter(), "added_node_labels": Counter()}
    for e in examples["train"]:
        for op in e.operations:
            if op.kind == "add_node":
                frequencies["added_node_labels"][op.after.label] += 1
            if op.after is not None and op.kind.endswith("node"):
                before = set(op.before.attributes) if op.before else set()
                frequencies["added_attributes"].update(sorted(set(op.after.attributes) - before))
    report["training_supervision_sparsity"] = {
        k: {
            "unique_values": len(v),
            "values_seen_once": sum(n == 1 for n in v.values()),
            "top20": sorted(v.items(), key=lambda x: (-x[1], x[0]))[:20],
        }
        for k, v in frequencies.items()
    }
    packet = {"sample_seed": SAMPLE_SEED, "model": "gnn", "seed": 42, "cases": [packets[e.id] for e in selected]}
    # Image checks establish exact image evidence for later qualitative review.
    for case in packet["cases"]:
        for role in ("source_image", "target_image"):
            assert sha256(ROOT / case[role]["path"]) == case[role]["sha256"]
    write(output / "quantitative.json", report)
    write(output / "cases.json", packet)
    print(
        {
            "output": str(output),
            "sample_ids": report["sample_ids"],
            "gnn42_all": report["models"]["gnn"]["42"]["all_67"],
            "sparsity": report["training_supervision_sparsity"],
        }
    )


if __name__ == "__main__":
    main()
