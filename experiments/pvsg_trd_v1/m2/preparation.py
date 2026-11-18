"""Versioned support filtering and grouped internal validation; raw data stays intact."""

import argparse
import copy
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from ..contracts import ROOT, normalize_targets, read_json, visibility_masks
from ..data import PREDICATE_IDS, rasterize_targets
from ..intervals import intervals_to_mask, mask_to_intervals
from ..validate_m0 import canonical_sha256, validate_manifest
from .download import sha256

POLICY_ID = "pvsg-gml-valid-support-v1"
POLICY = {
    "policy_id": POLICY_ID,
    "raw_annotation_unchanged": True,
    "all_400_video_ids_retained": True,
    "vocabulary": "frozen_57_predicates_no_alias_mapping",
    "unknown_predicate": "quarantine_row_like_upstream_ontology_filter",
    "unknown_track_or_self_pair": "quarantine_unrepresentable_row",
    "time": "intersect_integer_inclusive_intervals_with_existing_mask_timeline",
    "visibility": "intersect_both_gt_track_visibility_masks",
    "empty_support": "quarantine_row_with_explicit_ledger_before_M1_validation",
    "negative_supervision": "ignore_negative_labels_for_known_keys_with_invalid_time_or_empty_support_keep_valid_positives",
    "selection_basis": "upstream_loader_and_train_structural_audit_no_test_accuracy_tuning",
    "upstream_ontology_source": "https://github.com/LilyDaytoy/OpenPVSG/blob/b2850b980c272368215a754299986cb904d55820/utils/relation_matching.py",
    "benchmark_claim": "GML_GT_track_dataset_variant_not_original_PVSG_benchmark_reproduction",
}


def rasterize_adapted_targets(video, target, ignored_negative_keys, *, start=0, stop=None):
    """Use this supervision mask with the adapted dataset, for every baseline."""
    import numpy as np

    batch = rasterize_targets(video, target, start=start, stop=stop)
    mask = np.broadcast_to(batch["loss_mask"][..., None], batch["labels"].shape).copy()
    pairs = {tuple(pair): i for i, pair in enumerate(batch["pairs"])}
    for row in ignored_negative_keys:
        pair = row["subject_id"], row["object_id"]
        if pair in pairs:
            index, predicate = pairs[pair], row["predicate_id"]
            mask[index, :, predicate] &= batch["labels"][index, :, predicate] > 0
    batch["supervision_mask"] = mask
    return batch


def source_group(video_id, source):
    """Groups inferred from release IDs; not a claim of full session metadata."""
    if source == "ego4d":
        match = re.fullmatch(r"([0-9a-f-]{36})(?:_\d+)?", video_id)
        if not match:
            raise ValueError(f"Unrecognized Ego4D ID: {video_id}")
        key = match.group(1)
    elif source == "vidor":
        match = re.fullmatch(r"\d{4}_(\d+)", video_id)
        if not match:
            raise ValueError(f"Unrecognized VidOR ID: {video_id}")
        key = match.group(1)
    elif source == "epic_kitchen":
        match = re.fullmatch(r"(P\d+_\d+)(?:_\d+)?", video_id)
        if not match:
            raise ValueError(f"Unrecognized EPIC ID: {video_id}")
        key = match.group(1)
    else:
        raise ValueError(f"Unknown source: {source}")
    return f"{source}:{key}"


def internal_split(manifest, seed=17, fraction=0.15):
    validate_manifest(manifest, require_selected=True)
    groups = defaultdict(list)
    for video_id in manifest["outer_split"]["train_ids"]:
        groups[source_group(video_id, manifest["video_sources"][video_id])].append(video_id)
    validation = []
    for source in ("vidor", "epic_kitchen", "ego4d"):
        source_groups = {key: ids for key, ids in groups.items() if key.startswith(source + ":")}
        target = round(sum(map(len, source_groups.values())) * fraction)
        order = sorted(source_groups, key=lambda key: hashlib.sha256(f"{seed}:{key}".encode()).hexdigest())
        # Subset-sum chooses the closest source count while keeping whole groups.
        possible = {0: ()}
        for key in order:
            for count, selected in list(possible.items()):
                new_count = count + len(source_groups[key])
                if new_count not in possible:
                    possible[new_count] = selected + (key,)
        selected_count = min(possible, key=lambda count: (abs(count - target), count))
        validation.extend(video for key in possible[selected_count] for video in source_groups[key])
    result = copy.deepcopy(manifest)
    fit = sorted(set(manifest["outer_split"]["train_ids"]) - set(validation))
    result["split_id"] = f"pvsg-author-338-62-fit{len(fit)}-val{len(validation)}-seed{seed}-m2-v1"
    result["internal_validation"].update(state="locked", fit_ids=fit, validation_ids=sorted(validation))
    result["provenance"][
        "selection_note"
    ] = "Outer 338/62 chosen by user at M0; internal validation generated from train only using source-ID groups, seed 17 and approximately 15% per source."
    validate_manifest(result, require_selected=True)
    fit_groups = {source_group(v, result["video_sources"][v]) for v in fit}
    validation_groups = {source_group(v, result["video_sources"][v]) for v in validation}
    if fit_groups & validation_groups:
        raise ValueError("Source group crosses fit/validation")
    test_groups = {source_group(v, result["video_sources"][v]) for v in result["outer_split"]["heldout_ids"]}
    audit = {
        "algorithm": "source-stratified_hash_order_subset_sum_v1",
        "seed": seed,
        "target_fraction": fraction,
        "grouping_basis": "release_video_id_inference_not_complete_session_metadata",
        "group_by_video": {v: source_group(v, s) for v, s in result["video_sources"].items()},
        "fit_validation_shared_groups": [],
        "official_train_test_shared_source_groups": sorted((fit_groups | validation_groups) & test_groups),
        "counts": {
            split: dict(Counter(result["video_sources"][v] for v in ids))
            for split, ids in (("fit", fit), ("validation", validation), ("test", result["outer_split"]["heldout_ids"]))
        },
    }
    return result, audit


def adapt_targets(record, video):
    """Make all filtering visible in a ledger; never guess labels/time scaling."""
    rows, ledger, ignore = [], [], set()
    visible = visibility_masks(video)
    frames = video["num_frames"]
    for index, raw in enumerate(record["relations"]):
        entry = {"video_id": video["video_id"], "raw_row_index": index, "raw_relation": copy.deepcopy(raw)}
        if not isinstance(raw, list) or len(raw) != 4:
            raise ValueError("Unexpected raw relation structure; policy does not cover it")
        s, o, label, intervals = raw
        if label not in PREDICATE_IDS:
            entry.update(action="quarantine", reason="outside_frozen_ontology")
            ledger.append(entry)
            continue
        predicate = PREDICATE_IDS[label]
        if s == o or s not in visible or o not in visible:
            entry.update(action="quarantine", reason="self_pair_or_unknown_track")
            ledger.append(entry)
            continue
        clipped = []
        changed = False
        for start, end in intervals:
            if type(start) is not int or type(end) is not int or end < start:
                raise ValueError("Malformed interval; policy refuses to infer a correction")
            low, high = max(0, start), min(frames - 1, end)
            changed |= low != start or high != end
            if low <= high:
                clipped.append([low, high])
        support = intervals_to_mask(clipped, frames) & visible[s] & visible[o]
        if changed or not support.any():
            ignore.add((s, predicate, o))
        if not support.any():
            entry.update(action="quarantine", reason="no_covisible_support_in_mask_timeline")
            ledger.append(entry)
            continue
        normalized = mask_to_intervals(support)
        if changed or normalized != intervals:
            entry.update(
                action="intersect_support",
                reason="time_bounds" if changed else "visibility_or_interval_union",
                retained_intervals=normalized,
            )
            ledger.append(entry)
        rows.append({"subject_id": s, "object_id": o, "predicate_id": predicate, "intervals": normalized})
    target = normalize_targets(
        {
            "schema_version": 1,
            "kind": "relation_targets",
            "video_id": video["video_id"],
            "num_frames": frames,
            "relations": rows,
        },
        video,
    )
    return target, ledger, [{"subject_id": s, "predicate_id": p, "object_id": o} for s, p, o in sorted(ignore)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("data/external/pvsg_m2_cache_v4"))
    parser.add_argument("--annotation", type=Path, default=Path("data/external/pvsg_m0/pvsg.hf_7e5f1ec.json"))
    args = parser.parse_args()
    audit = read_json(args.cache / "media_audit.json")
    if audit["passed"] != 400 or audit["errors"]:
        raise ValueError("All 400 media inputs must pass audit before preparing targets")
    manifest = read_json(ROOT / "manifests/pvsg_author_release_338_62.lock.json")
    if sha256(args.annotation) != manifest["annotation"]["sha256"]:
        raise ValueError("Raw annotation identity mismatch")
    internal, group_audit = internal_split(manifest)
    records = {r["video_id"]: r for r in read_json(args.annotation)["data"]}
    inputs = read_json(args.cache / "inputs.json")
    if {v["video_id"] for v in inputs} != set(manifest["video_sources"]):
        raise ValueError("Input cohort mismatch")
    targets, ledger, ignore = [], [], {}
    for video in inputs:
        target, changes, ignored = adapt_targets(records[video["video_id"]], video)
        targets.append(target)
        ledger.extend(changes)
        ignore[video["video_id"]] = ignored
    fit = set(internal["internal_validation"]["fit_ids"])
    train_support = Counter(
        row["predicate_id"] for target in targets if target["video_id"] in fit for row in target["relations"]
    )
    report = {
        "policy": POLICY,
        "policy_sha256": canonical_sha256(POLICY),
        "raw_annotation_sha256": manifest["annotation"]["sha256"],
        "videos_retained": len(targets),
        "final_gt_tuples": sum(len(t["relations"]) for t in targets),
        "actions": dict(Counter((r["action"] + ":" + r["reason"]) for r in ledger)),
        "fit_tuple_support_by_predicate": [train_support[p] for p in range(57)],
        "statistics_split": "fit_only",
        "ignore_negative_keys": ignore,
        "ledger": ledger,
    }
    for filename, content in (
        ("targets.json", targets),
        ("annotation_adaptation.json", report),
        ("dataset.lock.json", internal),
        ("groups.json", group_audit),
    ):
        (args.cache / filename).write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "videos": len(targets),
                "gt_tuples": report["final_gt_tuples"],
                "actions": report["actions"],
                "split": group_audit["counts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
