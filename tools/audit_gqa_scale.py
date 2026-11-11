"""Audit GQA train/validation coverage and graph cost without opening test annotations.

Candidate training cohorts are nested prefixes of the frozen pilot hash order.
Label statistics describe those fixed cohorts; they never select their members.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from src.datasets.gqa import normalize_gqa
from src.knowledge.transe.core import sha256
from tools.prepare_level2_gqa import write_fixed


def ordered_train_ids(ids, seed=42):
    return sorted(ids, key=lambda i: (hashlib.sha256(f"gqa-pilot-v1:{seed}:train:{i}".encode()).hexdigest(), i))


def describe_record(record, normalization):
    predicates = Counter(r.predicate_id for r in record.relations)
    pairs = Counter((r.subject_id, r.object_id) for r in set(record.relations))
    objects = len(record.objects)
    candidates = objects * (objects - 1)
    positives = len(pairs)
    backgrounds = candidates - positives
    return {
        "image_id": record.image_id,
        "objects": objects,
        "candidate_pairs": candidates,
        "positive_pairs": positives,
        "background_pairs": backgrounds,
        "multi_predicate_pairs": sum(n > 1 for n in pairs.values()),
        "gt_annotation_rows": len(record.relations),
        "unique_gt_triplets": len(set(record.relations)),
        "sampled_training_pairs": positives + min(backgrounds, 3 * positives if positives else 32),
        "predicate_rows": predicates,
        "object_classes": {o.label_id for o in record.objects},
        "normalization": normalization,
    }


def distribution(values):
    values = np.asarray(values, dtype=np.int64)
    if not len(values):
        raise ValueError("Empty cohort")
    return {
        "min": int(values.min()),
        "mean": float(values.mean()),
        **{f"p{p}": float(np.percentile(values, p)) for p in (50, 90, 95, 99)},
        "max": int(values.max()),
        "total": int(values.sum()),
    }


def summarize(records, predicate_count, directional_ids):
    support, images, normalization = Counter(), Counter(), Counter()
    object_classes = set()
    for r in records:
        support.update(r["predicate_rows"])
        images.update(r["predicate_rows"].keys())
        normalization.update(r["normalization"])
        object_classes.update(r["object_classes"])
    total = sum(support.values())
    directional = sum(support[p] for p in directional_ids)
    counts = {
        "images": len(records),
        "images_without_gt": sum(not r["gt_annotation_rows"] for r in records),
        "images_without_candidates": sum(not r["candidate_pairs"] for r in records),
        "object_classes_observed": len(object_classes),
        "gt_annotation_rows": total,
        "directional_gt_rows": directional,
        "directional_fraction": directional / total if total else None,
        "non_directional_gt_rows": total - directional,
        "unique_gt_triplets": sum(r["unique_gt_triplets"] for r in records),
        "multi_predicate_pairs": sum(r["multi_predicate_pairs"] for r in records),
        "positive_pairs": sum(r["positive_pairs"] for r in records),
        "background_pairs": sum(r["background_pairs"] for r in records),
        "sampled_training_pairs": sum(r["sampled_training_pairs"] for r in records),
        "predicate_coverage": {
            "total_vocabulary": predicate_count,
            "with_gt_rows_at_least": {
                str(t): sum(support[p] >= t for p in range(1, predicate_count + 1)) for t in (1, 5, 20, 100, 1000)
            },
            "with_gt_images_at_least": {
                str(t): sum(images[p] >= t for p in range(1, predicate_count + 1)) for t in (1, 5, 20, 100)
            },
            "absent_ids": [p for p in range(1, predicate_count + 1) if not support[p]],
        },
        "objects": distribution([r["objects"] for r in records]),
        "candidates": distribution([r["candidate_pairs"] for r in records]),
        "large_graphs": {
            str(t): {
                "images": sum(r["objects"] >= t for r in records),
                "candidate_pairs": sum(r["candidate_pairs"] for r in records if r["objects"] >= t),
            }
            for t in (32, 64, 100, 128)
        },
        "largest_graphs": [
            {k: r[k] for k in ("image_id", "objects", "candidate_pairs", "gt_annotation_rows")}
            for r in sorted(records, key=lambda r: (-r["objects"], r["image_id"]))[:10]
        ],
        "normalization": dict(sorted(normalization.items())),
        "ordered_image_ids_sha256": hashlib.sha256(json.dumps([r["image_id"] for r in records]).encode()).hexdigest(),
        "cost_components": {
            "float16_clip_vectors_bytes": sum(r["objects"] for r in records) * 512 * 2,
            "largest_image_fp32_all_pair_logits_bytes": max(r["candidate_pairs"] for r in records)
            * (predicate_count + 1)
            * 4,
        },
    }
    return counts, support, images


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/gqa"))
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/level2/gqa-v1/manifest.json"))
    parser.add_argument("--pilot-cohort", type=Path, default=Path("reports/level2/gqa-pilot-v1-cohort.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/level2/gqa-scale-audit-v1.json"))
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    cohort = json.loads(args.pilot_cohort.read_text())
    if cohort["manifest_sha256"] != sha256(args.manifest):
        raise ValueError("Pilot and project manifest disagree")
    splits = manifest["splits"]
    sets = [set(splits[k]) for k in ("train", "val", "test")]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Overlapping project splits")
    if any(len(sets[i]) != len(splits[k]) for i, k in enumerate(("train", "val", "test"))):
        raise ValueError("Duplicate project IDs")
    train_order = ordered_train_ids(splits["train"], cohort["seed"])
    if train_order[: len(cohort["splits"]["train"])] != cohort["splits"]["train"]:
        raise ValueError("Pilot train is not prefix of proposed scale order")
    if not set(cohort["splits"]["val"]) <= sets[1]:
        raise ValueError("Pilot validation outside project validation")
    # Only official train is opened. Official val is the project test and stays unread.
    source = args.root / "sceneGraphs/train_sceneGraphs.json"
    if sha256(source) != manifest["annotation_sha256"][source.name]:
        raise ValueError("Training annotation hash mismatch")
    annotations = json.loads(source.read_text())
    vocab = manifest["vocabulary"]
    num_predicates = len(vocab["predicate_to_idx"])
    if set(vocab["predicate_to_idx"].values()) != set(range(1, num_predicates + 1)):
        raise ValueError("Expected contiguous predicate vocabulary")
    normalized = {}
    for split in ("train", "val"):
        for i, image_id in enumerate(splits[split]):
            record, stats = normalize_gqa(image_id, annotations[image_id], vocab)
            normalized[image_id] = describe_record(record, stats)
            if (i + 1) % 20000 == 0:
                print(f"[AUDIT] {split}: {i + 1}", flush=True)
    del annotations
    groups = {f"train_{n}": train_order[:n] for n in (512, 5000, 10000, 20000, 30000, 50000) if n < len(train_order)}
    groups["train_full"] = train_order
    groups["val_pilot"] = cohort["splits"]["val"]
    groups["val_full"] = splits["val"]
    summaries, counts, image_counts = {}, {}, {}
    directional_ids = [vocab["predicate_to_idx"][name] for name in ("to the left of", "to the right of")]
    for key, ids in groups.items():
        summaries[key], counts[key], image_counts[key] = summarize(
            [normalized[i] for i in ids], num_predicates, directional_ids
        )
    full_train, full_val = counts["train_full"], counts["val_full"]
    for key in groups:
        if not key.startswith("train"):
            continue
        missing = [p for p in full_val if not counts[key][p]]
        summaries[key]["validation_coverage_diagnostic"] = {
            "val_predicates_absent_from_this_train": sorted(missing),
            "val_gt_rows_for_those_predicates": sum(full_val[p] for p in missing),
            "note": "Diagnostic only; does not change cohort membership, vocabulary, validation or model selection metric",
        }
    rows = [
        {
            "id": p,
            "name": name,
            "annotation_rows": {key: counts[key][p] for key in groups},
            "images_with_gt": {key: image_counts[key][p] for key in groups},
        }
        for name, p in sorted(vocab["predicate_to_idx"].items(), key=lambda item: item[1])
    ]
    report = {
        "schema_version": 1,
        "scope": "train and validation annotation audit only",
        "manifest_sha256": sha256(args.manifest),
        "pilot_cohort_sha256": sha256(args.pilot_cohort),
        "source_sha256": {str(source): sha256(source)},
        "code_sha256": {path: sha256(Path(path)) for path in ("tools/audit_gqa_scale.py", "src/datasets/gqa.py")},
        "selection": "nested train prefixes of SHA256(gqa-pilot-v1:42:train:image_id); full project val unchanged",
        "quantiles": "numpy percentile, default linear interpolation",
        "test_annotation_opened": False,
        "test_model_evaluated": False,
        "training_performed": False,
        "proposed_cohorts_frozen_for_training": False,
        "summaries": summaries,
        "per_predicate": rows,
        "rare_full_train": {
            "zero_after_normalization": [r["name"] for r in rows if full_train[r["id"]] == 0],
            "one_to_four_rows": [r["name"] for r in rows if 1 <= full_train[r["id"]] < 5],
            "five_to_nineteen_rows": [r["name"] for r in rows if 5 <= full_train[r["id"]] < 20],
        },
        "limitations": [
            "Annotation-only audit; not new model results or image decode audit",
            "Candidate allocation estimates exclude activations, gradients, optimizer and allocator workspace",
            "Full validation support was inspected only for diagnostic reporting; test annotation not opened",
            "Vocabulary is fixed from raw project train; zero support after normalization remains in denominator",
        ],
    }
    write_fixed(args.report, report)
    print(f"[PASS] {len(normalized)} graphs; report={args.report}", flush=True)


if __name__ == "__main__":
    main()
