"""Pinned dataset and actual overall human verdicts; no fabricated axis ratings."""
from collections import Counter

from common.editing.operations import EditGraph, Operation, apply_operations, diff_aligned_graphs
from task2_graph_editing.magicbrush.sequence_model.codec import encode_actions, normalize_label
from task2_graph_editing.magicbrush.sequence_model.data import Example, load_examples
from common.io import read, sha256



def artifact_paths(config, root):
    paths = {}
    for name, spec in config["artifacts"].items():
        path = (root / spec["path"]).resolve()
        if not path.is_relative_to(root.resolve()) or sha256(path) != spec["sha256"]:
            raise ValueError(f"Pinned artifact changed: {name}")
        paths[name] = path
    return paths


def audit_splits(original, trainval, test):
    index = {r["id"]: r for r in original["records"]}
    if len(index) != len(original["records"]):
        raise ValueError("Duplicate original IDs")
    memberships, groups, sessions, pixels = {}, {}, {}, {}
    for split in ("train", "val", "test"):
        for ident in original["splits"][split]:
            if ident in memberships or ident not in index:
                raise ValueError("Duplicate or unknown split ID")
            memberships[ident] = split
    if set(memberships) != set(index):
        raise ValueError("Incomplete split coverage")
    for ident, r in index.items():
        split = r["split"]
        if memberships[ident] != split or r["official_split"] != "train":
            raise ValueError("Original split/provenance mismatch")
        for table, key in ((groups, r["source_group_id"]), (sessions, r["img_id"])):
            if table.setdefault(key, split) != split:
                raise ValueError("Source group/session leakage")
        for role in ("source_image", "target_image"):
            if pixels.setdefault(r[role]["pixel_sha256"], split) != split:
                raise ValueError("Image pixel leakage")
    for manifest, allowed in ((trainval, {"train", "val"}), (test, {"test"})):
        actual = {r["id"]: r for r in manifest["records"]}
        expected = {i: r for i, r in index.items() if r["split"] in allowed}
        if actual != expected or len(actual) != len(manifest["records"]):
            raise ValueError("Subset manifest differs from pinned original")
        for split in ("train", "val", "test"):
            wanted = original["splits"][split] if split in allowed else []
            if manifest["splits"][split] != wanted:
                raise ValueError("Subset split membership changed")
    return dict(Counter(r["split"] for r in index.values()))


def human_gate(manifest, labels, states, receipt):
    ids = manifest["review_ids"]
    cases = receipt["cases"]
    if (
        len(ids) != 20
        or len(set(ids)) != 20
        or len(cases) != 20
        or set(receipt["review_ids"]) != set(ids)
        or {r["id"] for r in cases} != set(ids)
    ):
        raise ValueError("Human review must cover exactly the fixed 20 IDs")
    if (
        receipt.get("manual_review_completed") is not True
        or receipt.get("evidence_source") != "user_reported_manual_review_in_conversation"
        or not receipt.get("reviewer")
        or not receipt.get("user_statements")
    ):
        raise ValueError("Missing actual human confirmation provenance")
    rows = {r["id"]: r for r in manifest["records"]}
    index = {r["id"]: r for r in labels["records"]}
    counts = Counter()
    for c in cases:
        ident = c["id"]
        row, label, state = rows[ident], index[ident], states[ident]
        if row["split"] != "train" or c["decision"] not in {"correct", "incorrect", "uncertain"}:
            raise ValueError("Invalid human verdict or non-train review")
        if (
            c["annotation_event_sha256"] != state["event_sha256"]
            or c["annotation_event_sha256"] != label["annotation_event_sha256"]
        ):
            raise ValueError("Human review event mismatch")
        for role in ("source", "target"):
            if c[role + "_image_sha256"] != row[role + "_image"]["sha256"]:
                raise ValueError("Human review image mismatch")
        a, b = (EditGraph.from_dict(state[k]) for k in ("source_graph", "target_graph"))
        a, b, ops = normalize_label(a, b, diff_aligned_graphs(a, b))
        if (
            a != EditGraph.from_dict(label["source_graph"])
            or b != EditGraph.from_dict(label["target_graph"])
            or ops != tuple(Operation.from_dict(o) for o in label["operations"])
        ):
            raise ValueError("Reviewed graph payload changed")
        counts[c["decision"]] += 1
    expected = {k: counts[k] for k in ("correct", "incorrect", "uncertain")}
    determinate = counts["correct"] + counts["incorrect"]
    fraction = counts["correct"] / determinate if determinate else None
    if receipt["decision_counts"] != expected or receipt["observed_correct_fraction"] != fraction:
        raise ValueError("Human aggregate differs from individual verdicts")
    return {
        "ready": determinate >= 20 and fraction is not None and fraction >= 0.8,
        "reviewer": receipt["reviewer"],
        "counts": expected,
        "observed_correct_fraction": fraction,
        "per_axis_ratings_invented": False,
        "scope": "overall correctness of fixed 20 train cases",
        "rejected_ids": [c["id"] for c in cases if c["decision"] != "correct"],
    }


def prepare_trainval(config, root):
    paths = artifact_paths(config, root)
    original, manifest, test = (read(paths[k]) for k in ("original_manifest", "trainval_manifest", "test_manifest"))
    raw_counts = audit_splits(original, manifest, test)
    gate = human_gate(
        manifest, read(paths["trainval_edits"]), read(paths["reviewed_states"]), read(paths["human_review"])
    )
    if not gate["ready"]:
        raise ValueError("Human review gate not satisfied")
    examples, loader = load_examples(
        paths["trainval_manifest"],
        paths["trainval_edits"],
        rejected=gate["rejected_ids"],
        max_action_tokens=config["max_action_tokens"],
    )
    loader["label_origin"] = "paired_image_AI_assisted_annotation_with_20_user_reviewed_train_examples"
    counts = {k: len(v) for k, v in examples.items()}
    if counts != config["expected_counts"]["trainval"] or sum(counts.values()) < 300 or not all(counts.values()):
        raise ValueError("Unexpected train/validation cohort size")
    return (
        examples,
        {
            "ready": True,
            "human_review": gate,
            "loader": loader,
            "raw_pair_counts": raw_counts,
            "test_targets_decoded": False,
        },
        paths,
    )


def load_heldout(paths, expected_count, max_tokens):
    """Only called by the evaluation stage after all selected checkpoints are frozen."""
    manifest, edits = read(paths["test_manifest"]), read(paths["test_edits"])
    if edits["dataset_manifest_sha256"] != sha256(paths["test_manifest"]):
        raise ValueError("Test edits/manifest mismatch")
    index = {r["id"]: r for r in manifest["records"]}
    seen, examples = set(), []
    for label in edits["records"]:
        ident = label["id"]
        if ident not in index or ident in seen:
            raise ValueError("Unknown or duplicate test label")
        seen.add(ident)
        row = index[ident]
        if any(label[k] != row[k] for k in ("split", "source_group_id", "instruction")) or row["split"] != "test":
            raise ValueError("Test metadata differs from original")
        if (
            label["eligible_for_training"] is not False
            or label["eligible_for_evaluation"] is not True
            or label["status"] != "candidate"
            or label["problems"]
        ):
            raise ValueError("Unexpected heldout eligibility")
        a, b = (EditGraph.from_dict(label[k]) for k in ("source_graph", "target_graph"))
        ops = tuple(Operation.from_dict(o) for o in label["operations"])
        a, b, ops = normalize_label(a, b, ops)
        if (
            not 1 <= len(ops) <= 12
            or max(len(a.nodes), len(b.nodes)) > 20
            or max(len(a.edges), len(b.edges)) > 60
            or len(encode_actions(ops)) - 1 > max_tokens
            or apply_operations(a, ops) != b
        ):
            raise ValueError("Heldout label violates frozen budgets")
        examples.append(Example(ident, "test", row["source_group_id"], row["instruction"], a, b, ops))
    if len(examples) != expected_count:
        raise ValueError("Unexpected heldout label count")
    examples.sort(key=lambda e: e.id)
    return examples, {
        "original_test_pairs": len(index),
        "evaluated_pairs": len(examples),
        "deferred_pairs": len(index) - len(examples),
        "evaluated_fraction": len(examples) / len(index),
        "excluded_ids": sorted(set(index) - seen),
        "exclusion_basis": "frozen annotation decisions before model runs",
    }
