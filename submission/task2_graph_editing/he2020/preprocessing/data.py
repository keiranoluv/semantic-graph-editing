"""Dataset-specific provenance loader, sharing the existing Task B Example/codec.

The frozen MagicBrush loader requires images and official-train provenance;
inventing pixels or claiming crowdsourced dev is train would violate its contract.
This versioned entry point keeps those contracts intact and returns identical
Example/model_input objects for the graph predictor.
"""

from collections import Counter
from itertools import permutations
from pathlib import Path

from common.editing.operations import EditGraph, Operation, apply_operations
from task2_graph_editing.magicbrush.sequence_model.codec import decode_actions, encode_actions, normalize_label
from task2_graph_editing.magicbrush.sequence_model.data import Example
from task2_graph_editing.he2020.preprocessing.alignment import validate_raw
from common.io import digest, read, sha256
from task2_graph_editing.he2020.preprocessing.dataset import ARCHIVE_SHA256


def source_fingerprint(raw):
    """ID/order independent graph group; no target information, max seven nodes."""
    nodes, edges = validate_raw(raw)
    if len(nodes) > 7:
        raise ValueError("fingerprint_search_budget")
    # Only permute within equal labels; all permutations across labels would
    # produce a larger label tuple and cannot be the lexicographic minimum.
    groups = {}
    for k, v in nodes.items():
        groups.setdefault(v, []).append(k)
    orders = [()]
    for label in sorted(groups):
        orders = [prefix + order for prefix in orders for order in permutations(groups[label])]
    encodings = []
    for order in orders:
        ids = {k: i for i, k in enumerate(order)}
        triples = []
        for (u, v), p in edges.items():
            a, b = ids[u], ids[v]
            if not raw["directed"]:
                a, b = sorted((a, b))
            triples.append((a, p, b))
        encodings.append((tuple(nodes[k] for k in order), tuple(sorted(triples))))
    return digest([raw["directed"], min(encodings)])


def leakage_filter(records):
    """Retain evidence; flag repeats and dev sources seen in train explicitly."""
    groups, pairs, excluded = {}, {}, {}
    # Official train wins conflicts. Never move a dev example into train.
    for row in sorted(records, key=lambda r: (r["split"] != "train", r["original_record_index"])):
        if row.get("raw_source") is None:
            continue
        try:
            group = source_fingerprint(row["raw_source"])
        except ValueError:
            continue
        row["source_group_id"] = group
        pair = digest([group, " ".join(row["instruction"].split()).casefold()])
        row["source_instruction_sha256"] = pair
        if group in groups and groups[group] != row["split"]:
            excluded[row["id"]] = "duplicate_source_graph_across_official_splits"
        elif pair in pairs:
            excluded[row["id"]] = "duplicate_source_instruction_pair"
        else:
            groups[group], pairs[pair] = row["split"], row["id"]
    return excluded


def checked_inputs(output):
    output = Path(output)
    manifest, edits = read(output / "manifest.json"), read(output / "edits.json")
    if manifest["archive_sha256"] != ARCHIVE_SHA256 or edits["dataset_manifest_sha256"] != sha256(
        output / "manifest.json"
    ):
        raise ValueError("manifest_provenance_mismatch")
    for name, checksum in manifest["inputs"].items():
        if Path(name).name != name or sha256(output / name) != checksum:
            raise ValueError("input_evidence_changed")
    records = manifest["records"]
    index, labels = {r["id"]: r for r in records}, {r["id"]: r for r in edits["records"]}
    if len(index) != len(records) or len(labels) != len(edits["records"]) or index.keys() != labels.keys():
        raise ValueError("duplicate_missing_or_unknown_sample_ids")
    memberships, groups, pairs = {}, {}, {}
    raw = {r["id"]: r for r in read(output / "raw-records.json")}
    if raw.keys() != index.keys():
        raise ValueError("raw_cohort_differs")
    for split in ("train", "val", "test"):
        if split == "test" and manifest["splits"][split]:
            raise ValueError("test_labels_forbidden")
        for identifier in manifest["splits"][split]:
            if identifier in memberships or identifier not in index:
                raise ValueError("overlapping_or_unknown_membership")
            memberships[identifier] = split
    if memberships.keys() != index.keys():
        raise ValueError("incomplete_split_membership")
    for identifier, row in index.items():
        expected_official = "train" if row["split"] == "train" else "dev"
        if row["official_split"] != expected_official or memberships[identifier] != row["split"]:
            raise ValueError("official_split_changed")
        label = labels[identifier]
        original = raw[identifier]
        for k in ("id", "split", "official_split", "instruction", "original_record_index"):
            if row[k] != label[k] or row[k] != original[k]:
                raise ValueError("record_metadata_changed")
        if label["raw_source"] != original["raw_source"] or label["raw_target"] != original["raw_target"]:
            raise ValueError("raw_graph_evidence_changed")
        if row["raw_evidence_sha256"] != digest(original):
            raise ValueError("raw_evidence_hash_changed")
        # Bind normalized graphs/actions to the raw evidence, not merely to a
        # self-consistent edited JSON. Re-derive offline labels without changing
        # artifacts; this also verifies that source IDs never depend on target.
        from task2_graph_editing.he2020.preprocessing.compiler import compile_record

        expected = compile_record(original)
        for key in ("conversion_policy", "source_graph", "target_graph", "operations", "alignment", "lossy_conversion"):
            if digest(label.get(key)) != digest(expected.get(key)):
                raise ValueError("normalized_label_differs_from_raw_derivation")
        if label.get("status") != "candidate":
            continue
        group = source_fingerprint(original["raw_source"])
        pair = digest([group, " ".join(row["instruction"].split()).casefold()])
        if row["source_group_id"] != group or row["source_instruction_sha256"] != pair:
            raise ValueError("source_group_identity_changed")
        if groups.setdefault(group, row["split"]) != row["split"] or pair in pairs:
            raise ValueError("source_or_instruction_leakage")
        pairs[pair] = identifier
    return manifest, index, labels


def load_examples(output, *, rejected=(), max_action_tokens=2048):
    manifest, index, labels = checked_inputs(output)
    examples, excluded, long_validation = {"train": [], "val": []}, Counter(), []
    for split in examples:
        for identifier in manifest["splits"][split]:
            row, label = index[identifier], labels[identifier]
            if identifier in rejected:
                excluded["human_incorrect_or_uncertain"] += 1
                continue
            if label["status"] != "candidate" or label["problems"] or not label["eligible_for_training"]:
                excluded["not_candidate"] += 1
                continue
            source, target = (EditGraph.from_dict(label[k]) for k in ("source_graph", "target_graph"))
            ops = tuple(Operation.from_dict(o) for o in label["operations"])
            if not 1 <= len(ops) <= 12 or len(source.nodes) > 20 or len(source.edges) > 60:
                raise ValueError("candidate_budget_violation")
            if apply_operations(source, ops) != target or normalize_label(source, target, ops) != (source, target, ops):
                raise ValueError("noncanonical_or_invalid_label")
            tokens = encode_actions(ops)
            if decode_actions(tokens[1:], source) != ops:
                raise ValueError("codec_roundtrip_changed")
            if len(tokens) - 1 > max_action_tokens:
                if split == "train":
                    excluded["train_action_sequence_too_long"] += 1
                    continue
                long_validation.append(identifier)
            examples[split].append(
                Example(identifier, split, row["source_group_id"], row["instruction"], source, target, ops)
            )
    return examples, {
        "examples": {s: len(rows) for s, rows in examples.items()},
        "excluded": dict(excluded),
        "validation_labels_exceeding_decode_budget": long_validation,
        "test_labels_decoded": False,
        "model_input": "existing_Example.model_input_source_and_instruction_only",
        "label_origin": "official_caption_derived_graphs_not_image_verified_GT",
        "magicbrush_image_manifest_compatible": False,
        "task_b_Example_codec_compatible": True,
    }
