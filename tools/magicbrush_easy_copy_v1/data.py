"""Versioned CPU export, paired before/after provenance and all-val evaluation."""
from dataclasses import replace

from tools.edit_model_v1.codec import normalize_label
from tools.emnlp2020_graph_modification_v1.dataset import read, sha256, write
from tools.magicbrush_compact_v1.build import example_dict, statistics
from tools.magicbrush_experiment_v3.pipeline import preflight as original_preflight
from tools.magicbrush_experiment_v3.vocabulary import build_vocabulary, coverage
from tools.prepare_magicbrush import ROOT, digest

from .candidates import candidates
from .projection import project_example


def prepare(config):
    parent, original, _, parent_identity, _ = original_preflight(ROOT / config["parent_config"])
    review = read(ROOT / config["review"]["path"])
    if sha256(ROOT / config["review"]["path"]) != config["review"]["sha256"]:
        raise ValueError("Visual review changed")
    excluded = {r["id"] for r in review["cases"] if r["decision"] == "quarantine_train"}
    if excluded != set(config["excluded_train_ids"]):
        raise ValueError("Train quarantine differs from pinned visual review")
    if excluded - {e.id for e in original["train"]}:
        raise ValueError("Quarantine can affect train only")
    for split in original:
        if any(e.split != split for e in original[split]):
            raise ValueError("Split mismatch")
    manifest = read(ROOT / parent["artifacts"]["original_manifest"]["path"])
    images = {r["id"]: r for r in manifest["records"]}
    for row in review["cases"]:
        for role in ("source_image", "target_image"):
            spec = images[row["id"]][role]
            if spec["sha256"] != row[role + "_sha256"] or sha256(ROOT / spec["path"]) != spec["sha256"]:
                raise ValueError("Reviewed image changed")
    examples, records, sidecar = {"train": [], "val": []}, [], []
    for split, items in original.items():
        for e in items:
            projected, audit = project_example(e)
            again, _ = project_example(projected)
            if projected != again or projected.source != e.source:
                raise ValueError("Non-idempotent projection or source changed")
            source, target, operations = normalize_label(projected.source, projected.target, projected.operations)
            normalized = replace(projected, source=source, target=target, operations=operations)
            status = "quarantined" if e.id in excluded else "included"
            records.append({**example_dict(normalized), "status": status, "original_sha256": digest(example_dict(e))})
            sidecar.append({"id": e.id, "original": example_dict(e), **audit})
            if status == "included":
                examples[split].append(normalized)
    if [e.id for e in examples["val"]] != [e.id for e in original["val"]]:
        raise AssertionError("Validation filtering forbidden")
    if len(examples["train"]) + len(examples["val"]) < 300:
        raise ValueError("Fewer than 300 development pairs")
    reviewed = {c["id"] for c in review["cases"]}
    if any(r["id"] not in reviewed for r in sidecar if r["omitted_new_node_appearance"]):
        raise ValueError("A simplified pair lacks visual review")
    vocab = build_vocabulary(examples["train"])
    context_attribute_count = len(vocab["attributes"])
    # Existing attributes are retained/deleted by source-copy decisions. Only
    # values actually inserted by TRAIN operations need output prototypes.
    delta_attributes = sorted(
        {
            a
            for e in examples["train"]
            for op in e.operations
            if op.kind in ("add_node", "replace_node")
            for a in set(op.after.attributes) - set(op.before.attributes if op.before else ())
        }
    )
    vocab["attributes"] = delta_attributes or ["visible"]
    vocab["attribute_fallback_if_training_has_no_attributes"] = not delta_attributes
    vocab[
        "attribute_vocabulary_scope"
    ] = "train operation insertions only; arbitrary existing attributes copied from source"
    copy_coverage = {}
    for split, items in examples.items():
        added = [(e, op) for e in items for op in e.operations if op.kind == "add_node"]
        copy_coverage[split] = {
            "added_nodes": len(added),
            "train_vocabulary_names": sum(op.after.label in vocab["labels"] for e, op in added),
            "instruction_copyable_names": sum(op.after.label in candidates(e.instruction) for e, op in added),
            "vocabulary_or_copy_names": sum(
                op.after.label in vocab["labels"] or op.after.label in candidates(e.instruction) for e, op in added
            ),
            "oov_copyable": [
                {"id": e.id, "name": op.after.label}
                for e, op in added
                if op.after.label not in vocab["labels"] and op.after.label in candidates(e.instruction)
            ],
        }
    report = {
        "schema": "magicbrush-easy-copy-v1",
        "parent_identity_sha256": digest(parent_identity),
        "schema_design": "Known train and validation failure cases informed this development revision; no test labels/images opened",
        "projection": "Original source and all existing target nodes retained; omit unspecified appearance only from unique new nodes in guarded plain additions; preserve every edge and operation kind",
        "split_statistics": {
            s: {"original": statistics(original[s]), "new": statistics(examples[s])} for s in original
        },
        "quarantined_train_ids": sorted(excluded),
        "all_validation_ids_retained": True,
        "omitted_appearance_pairs": sum(
            bool(r["omitted_new_node_appearance"]) for r in sidecar if r["id"] not in excluded
        ),
        "omitted_appearance_nodes": sum(
            len(r["omitted_new_node_appearance"]) for r in sidecar if r["id"] not in excluded
        ),
        "omitted_attribute_values": sum(
            len(n["attributes"]) for r in sidecar if r["id"] not in excluded for n in r["omitted_new_node_appearance"]
        ),
        "vocabulary_sizes": {k: len(vocab[k]) for k in ("labels", "attributes", "predicates")},
        "copy_name_coverage": copy_coverage,
        "validation_payload_coverage_without_copy": coverage(examples["val"], vocab),
        "review": {
            "assistant_visual_pairs": len(review["cases"]),
            "selection": review["selection"],
            "new_human_confirmation": False,
        },
        "attribute_prototype_reduction": {
            "all_context_values": context_attribute_count,
            "train_inserted_values": len(vocab["attributes"]),
            "training_added_values_dropped": 0,
        },
        "all_simplified_pairs_visually_reviewed": all(
            r["id"] in {c["id"] for c in review["cases"]} for r in sidecar if r["omitted_new_node_appearance"]
        ),
        "old_human_review_scope": "20/20 original user-reviewed labels; does not certify every new projected label",
        "test_targets_decoded": False,
        "remaining_limitations": [
            "Original property vocabulary remains rich; no dynamic attribute/relation copying",
            "Observed incidental structural changes still supervised",
            "Copy requires matching instruction span; synonyms are not inferred",
            "Assistant visual review is a sample, not a full-dataset reliability estimate",
        ],
    }
    return examples, vocab, records, sidecar, report


def export(config, identity, records, sidecar, report):
    output = ROOT / config["dataset_output"]
    write(output / "identity.json", identity)
    write(output / "trainval-edits.json", {"identity_sha256": digest(identity), "records": records})
    write(output / "detail-sidecar.json", {"identity_sha256": digest(identity), "records": sidecar})
    write(output / "vocabulary-train.json", identity["vocabulary"])
    write(ROOT / config["reports"] / "data-audit.json", report)
    write(
        ROOT / config["reports"] / "dataset-manifest.json",
        {
            "identity_sha256": digest(identity),
            "files": {str(p.relative_to(ROOT)): sha256(p) for p in sorted(output.glob("*.json"))},
        },
    )
