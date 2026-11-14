"""CPU-only, immutable train/validation export of a frozen coarse attribute policy."""
import json
import random
from collections import Counter
from pathlib import Path

from src.editing.operations import KINDS
from tools.edit_model_v1.codec import normalize_label
from tools.emnlp2020_graph_modification_v1.dataset import read, sha256, write
from tools.magicbrush_experiment_v3.pipeline import preflight as parent_preflight
from tools.magicbrush_experiment_v3.pipeline import source_hashes as parent_hashes
from tools.magicbrush_experiment_v3.vocabulary import build_vocabulary, coverage
from tools.prepare_magicbrush import ROOT, digest

from .projection import project_example, project_graph

CONFIG = ROOT / "configs/level2/magicbrush-compact-v1.json"


def source_hashes():
    return {
        **parent_hashes(),
        **{str(p.relative_to(ROOT)): sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
    }


def example_dict(e):
    return {
        "id": e.id,
        "split": e.split,
        "source_group_id": e.group,
        "instruction": e.instruction,
        "source_graph": e.source.to_dict(),
        "target_graph": e.target.to_dict(),
        "operations": [o.to_dict() for o in e.operations],
    }


def statistics(examples):
    attributes = Counter(a for e in examples for g in (e.source, e.target) for n in g.nodes for a in n.attributes)
    nodes = sum(len(g.nodes) for e in examples for g in (e.source, e.target))
    operations = Counter(o.kind for e in examples for o in e.operations)
    return {
        "pairs": len(examples),
        "nonempty_edit_pairs": sum(bool(e.operations) for e in examples),
        "distinct_attributes": len(attributes),
        "attribute_occurrences": sum(attributes.values()),
        "attributes_per_node": sum(attributes.values()) / nodes if nodes else 0,
        "operations": sum(operations.values()),
        "operation_support": {k: operations[k] for k in KINDS},
        "distinct_labels": len({n.label for e in examples for g in (e.source, e.target) for n in g.nodes}),
        "distinct_predicates": len({x.predicate for e in examples for g in (e.source, e.target) for x in g.edges}),
    }


def build():
    config = read(CONFIG)
    lock = read(CONFIG.with_suffix(".lock.json"))
    if lock != {"config_sha256": sha256(CONFIG), "source_sha256": source_hashes()}:
        raise ValueError("Compact release lock changed")
    parent_path, policy_path = (ROOT / config[k] for k in ("parent_config", "policy"))
    if sha256(parent_path) != config["parent_config_sha256"] or sha256(policy_path) != config["policy_sha256"]:
        raise ValueError("Pinned parent or policy changed")
    parent, original, parent_report, _, _ = parent_preflight(parent_path)
    policy = read(policy_path)
    output, reports = (ROOT / config[k] for k in ("output", "reports"))
    if (
        output != ROOT / "data/processed/level2/magicbrush-compact-v1"
        or reports != ROOT / "reports/level2/magicbrush-compact-v1"
    ):
        raise ValueError("Expected isolated compact output paths")
    identity = {
        "schema": policy["schema"],
        "config_sha256": sha256(CONFIG),
        "release_lock_sha256": sha256(CONFIG.with_suffix(".lock.json")),
        "policy_sha256": sha256(policy_path),
        "parent_config_sha256": sha256(parent_path),
        "original_trainval_edits_sha256": parent["artifacts"]["trainval_edits"]["sha256"],
        "original_split_manifest_sha256": parent["artifacts"]["original_manifest"]["sha256"],
        "source_hashes": source_hashes(),
    }
    write(output / "identity.json", identity)
    projected, records, details, impacts = {}, [], [], []
    for split in ("train", "val"):
        projected[split] = []
        for e in original[split]:
            compact, audit = project_example(e, policy)
            for graph in (compact.source, compact.target):
                if project_graph(graph, policy)[0] != graph:
                    raise ValueError("Projection is not idempotent: " + e.id)
            if not audit["flags"]:
                normalize_label(compact.source, compact.target, compact.operations)
            projected[split].append(compact)
            records.append(
                {**example_dict(compact), "original_label_sha256": digest(example_dict(e)), "flags": audit["flags"]}
            )
            details.append({"id": e.id, "original": example_dict(e), **audit})
            impacts.append(
                {
                    "id": e.id,
                    "split": split,
                    "before_operations": len(e.operations),
                    "after_operations": len(compact.operations),
                    "removed_operations": audit["removed_operations"],
                    "flags": audit["flags"],
                }
            )
    # ALL pairs remain present, including possible no-op/ambiguous cases in val.
    for split in original:
        if [(e.id, e.group, e.instruction) for e in original[split]] != [
            (e.id, e.group, e.instruction) for e in projected[split]
        ]:
            raise ValueError("Cohort or source groups changed")
    vocabulary = build_vocabulary(projected["train"])
    write(output / "trainval-edits.json", {"identity_sha256": digest(identity), "records": records})
    write(output / "detail-sidecar.json", {"identity_sha256": digest(identity), "records": details})
    write(output / "vocabulary-train.json", vocabulary)
    summary = {
        "schema": policy["schema"],
        "identity_sha256": digest(identity),
        "scope": "Coarse attribute labels on existing accepted train/validation pairs; test labels remain unopened",
        "gpu_jobs_started": False,
        "test_targets_decoded": False,
        "parent_human_review": {
            "receipt_sha256": parent["artifacts"]["human_review"]["sha256"],
            "scope": "Original labels only. No new human verdicts inferred for compact labels.",
        },
        "schema_design_split": "train only; policy locked before validation projection",
        "splits": {
            split: {"before": statistics(original[split]), "after": statistics(projected[split])} for split in original
        },
        "flags": dict(Counter(flag for row in records for flag in row["flags"])),
        "flagged_cases": [{"id": row["id"], "flags": row["flags"]} for row in records if row["flags"]],
        "all_pair_ids_groups_instructions_and_split_memberships_preserved": True,
        "all_node_ids_labels_and_edge_payloads_preserved": True,
        "discarded_details_recoverable_in_sidecar": True,
        "projected_transaction_and_idempotence_checks_passed": True,
        "vocabulary_sizes": {k: len(vocabulary[k]) for k in ("labels", "attributes", "predicates")},
        "validation_payload_vocabulary_coverage": coverage(projected["val"], vocabulary),
        "removed_replace_node_operations": sum(len(row["removed_operations"]) for row in impacts),
        "pairs_with_removed_replace_operations": sum(bool(row["removed_operations"]) for row in impacts),
        "training_launched": False,
        "heldout_plan": "Apply the exact locked policy during a separate heldout evaluation stage; never fit aliases from test.",
        "metric_scope": "A different annotation protocol. Scores are not directly comparable to full-detail v3; both models need matched compact-label evaluation.",
    }
    write(reports / "impact.json", summary)
    write(reports / "case-impact.json", impacts)
    # Seeded random train cohort, independent of outcomes; retain before/after ops.
    train = sorted(original["train"], key=lambda e: e.id)
    sample = sorted(random.Random(config["review_seed"]).sample(train, config["review_size"]), key=lambda e: e.id)
    by_id = {row["id"]: row for row in details}
    compact_by_id = {row["id"]: row for row in records}
    write(
        reports / "review-20-train.json",
        {
            "seed": config["review_seed"],
            "selection": "random 20 train examples, before any filtering; not a new human review receipt",
            "cases": [
                {
                    "id": e.id,
                    "instruction": e.instruction,
                    "original": by_id[e.id]["original"],
                    "compact": compact_by_id[e.id],
                    "removed_operations": by_id[e.id]["removed_operations"],
                }
                for e in sample
            ],
        },
    )
    mappings = Counter(
        a for e in original["train"] for g in (e.source, e.target) for n in g.nodes for a in n.attributes
    )
    from .projection import project_attribute

    write(
        reports / "train-attribute-mapping.json",
        [
            {
                "original": a,
                "occurrences": count,
                "core": list(project_attribute(a, policy)[0]),
                "rule": project_attribute(a, policy)[1],
            }
            for a, count in sorted(mappings.items())
        ],
    )
    artifacts = [
        output / name
        for name in ("identity.json", "trainval-edits.json", "detail-sidecar.json", "vocabulary-train.json")
    ]
    artifacts += [
        reports / name
        for name in ("impact.json", "case-impact.json", "review-20-train.json", "train-attribute-mapping.json")
    ]
    write(
        reports / "manifest.json",
        {"identity_sha256": digest(identity), "files": {str(p.relative_to(ROOT)): sha256(p) for p in artifacts}},
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "report": str(reports / "impact.json"),
                "splits": summary["splits"],
                "flags": summary["flags"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return summary


if __name__ == "__main__":
    build()
