"""Freeze a train-only triple vocabulary and evaluate nine fixed fusion checkpoints on validation."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import torch

from scripts.evaluate_level1 import load_model
from scripts.pilot_mlp import resolve_device
from scripts.train_mlp import make_dataset
from src.datasets.batching import graph_batches
from src.evaluation.reference_predcls import NG_PROTOCOL, PROTOCOL as GC_PROTOCOL, REFERENCE_COMMIT
from src.evaluation.reference_predcls import evaluate_reference_image, summarize_reference
from src.evaluation.zero_shot import PROTOCOL, class_triplets, evaluate_zero_shot_image, summarize_zero_shot
from src.features.visual import load_visual_manifest
from src.graph.gnn_batching import collate_gnn_graphs, forward_gat_batch
from src.knowledge.transe.core import sha256
from tools.analyze_level1_long_tail import DATA_FILES, SEEDS, encode, stats, write_fixed
from tools.compare_reference_protocols import checked_reports, compare_reports

CONDITIONS = ("baseline", "pair_real", "pair_shuffled")


def raw_class_triples(h5, rows):
    """Read raw annotations, retaining duplicates and all predicates on a pair."""
    if len(set(rows)) != len(rows):
        raise ValueError("Repeated split image rows")
    for row in rows:
        if not 0 <= row < len(h5["img_to_first_rel"]):
            raise ValueError("Split row outside HDF5")
        first = int(np.asarray(h5["img_to_first_rel"][row]).item())
        last = int(np.asarray(h5["img_to_last_rel"][row]).item())
        if first == last == -1:
            yield []
            continue
        if first < 0 or last < first or last >= len(h5["predicates"]):
            raise ValueError("Invalid annotation range")
        pairs = h5["relationships"][first : last + 1]
        predicates = h5["predicates"][first : last + 1].reshape(-1)
        # Endpoints are global HDF5 object rows. Slice the containing image once.
        first_box = int(np.asarray(h5["img_to_first_box"][row]).item())
        last_box = int(np.asarray(h5["img_to_last_box"][row]).item())
        if first_box < 0 or last_box < first_box or last_box >= len(h5["labels"]):
            raise ValueError("Invalid object range")
        if ((pairs < first_box) | (pairs > last_box)).any():
            raise ValueError("Annotation endpoint outside image")
        labels = h5["labels"][first_box : last_box + 1].reshape(-1)
        triples = [
            (int(labels[s - first_box]), int(p), int(labels[o - first_box]))
            for (s, o), p in zip(pairs, predicates, strict=True)
        ]
        if any(not (1 <= s <= 150 and 1 <= o <= 150 and 1 <= p <= 50) for s, p, o in triples):
            raise ValueError("Class IDs outside VG150 vocabulary")
        yield triples


def seen_from_rows(h5, train_rows):
    seen, annotation_count = set(), 0
    for triples in raw_class_triples(h5, train_rows):
        seen.update(triples)
        annotation_count += len(triples)
    return seen, annotation_count


def prepare_cohort(root, output):
    splits = json.loads((root / "splits.json").read_text())
    if len(splits["train"]) != 57723 or len(splits["val"]) != 5000:
        raise ValueError("Expected full frozen VG150 train/validation splits")
    if set(splits["train"]) & (set(splits["val"]) | set(splits["test"])) or set(splits["val"]) & set(splits["test"]):
        raise ValueError("Overlapping image splits")
    hashes = {file: sha256(root / file) for file in DATA_FILES}
    with h5py.File(root / "VG-SGG-with-attri.h5", "r") as h5:
        seen, count = seen_from_rows(h5, splits["train"])
    vocabulary = {
        "schema_version": 1,
        "triple_order": ["subject_class", "predicate", "object_class"],
        "source_split": "train",
        "train_images": len(splits["train"]),
        "train_annotations": count,
        "policy": "all_raw_annotations_including_duplicates_and_multilabel_pairs_before_training_sampler",
        "unique_seen_triples": len(seen),
        "data_sha256": hashes,
        "triples": sorted(seen),
    }
    # Write the vocabulary before inspecting validation relations or model scores.
    vocabulary_path = output / "train-seen-triplets.json"
    write_fixed(vocabulary_path, encode(vocabulary))
    images, zero_counts, all_counts = [], Counter(), Counter()
    with h5py.File(root / "VG-SGG-with-attri.h5", "r") as h5:
        for i, triples in enumerate(raw_class_triples(h5, splits["val"])):
            unseen = [t for t in triples if t not in seen]
            all_counts.update(triples)
            zero_counts.update(unseen)
            images.append(
                {
                    "validation_index": i,
                    "hdf5_row": splits["val"][i],
                    "gt_annotations": len(triples),
                    "zero_shot_annotations": len(unseen),
                }
            )
    train_subjects = {t[0] for t in seen}
    train_objects = {t[2] for t in seen}
    train_predicates = {t[1] for t in seen}
    cohort = {
        "schema_version": 1,
        "split": "val",
        "images_evaluated": len(images),
        "train_vocabulary_sha256": sha256(vocabulary_path),
        "data_sha256": hashes,
        "gt_annotations": sum(all_counts.values()),
        "zero_shot_annotations": sum(zero_counts.values()),
        "images_with_zero_shot_gt": sum(i["zero_shot_annotations"] > 0 for i in images),
        "unique_zero_shot_triples": len(zero_counts),
        "all_zero_shot_components_seen_in_train_in_corresponding_roles": all(
            s in train_subjects and p in train_predicates and o in train_objects for s, p, o in zero_counts
        ),
        "zero_shot_gt_by_predicate": dict(
            sorted(Counter(p for (_, p, _), n in zero_counts.items() for _ in range(n)).items())
        ),
        "zero_shot_triples": [{"triple": t, "annotations": n} for t, n in sorted(zero_counts.items())],
        "images": images,
    }
    write_fixed(output / "validation-cohort.json", encode(cohort))
    return seen, vocabulary, cohort


def source_hashes():
    return {
        p: sha256(Path(p))
        for p in (
            "tools/level1_zero_shot.py",
            "src/evaluation/zero_shot.py",
            "src/evaluation/reference_predcls.py",
            "scripts/evaluate_level1.py",
            "src/graph/gnn_batching.py",
            "src/models/gat.py",
            "src/models/gat_kg_fusion.py",
            "tools/analyze_level1_long_tail.py",
            "tools/compare_reference_protocols.py",
        )
    }


def verify_report(row, entry, definition_hashes, code_hashes, cohort):
    if not (
        row["protocol"] == PROTOCOL
        and row["condition"] == entry["condition"]
        and row["seed"] == entry["seed"]
        and row["checkpoint_sha256"] == entry["checkpoint_sha256"]
        and row["definitions"] == definition_hashes
        and row["evaluator_sha256"] == code_hashes
        and row["images_evaluated"] == 5000
        and row["split"] == "val"
    ):
        raise ValueError("Incompatible existing zero-shot report")
    images = row["zero_shot_images"]
    expected_support = {
        r["validation_index"]: r["zero_shot_annotations"] for r in cohort["images"] if r["zero_shot_annotations"]
    }
    if (
        len(images) != len(expected_support)
        or {r["validation_index"]: r["support"] for r in images} != expected_support
    ):
        raise ValueError("Zero-shot per-image support differs")
    for mode in ("GC", "NG"):
        summary = row["zero_shot"][mode]
        if (
            summary["images_with_zero_shot_gt"] != cohort["images_with_zero_shot_gt"]
            or summary["zero_shot_gt_annotations"] != cohort["zero_shot_annotations"]
        ):
            raise ValueError("Zero-shot support differs from frozen cohort")
        if {int(c): n for c, n in summary["zero_shot_gt_by_predicate"].items()} != {
            int(c): n for c, n in cohort["zero_shot_gt_by_predicate"].items()
        }:
            raise ValueError("Zero-shot predicate support differs")
        prefix = "" if mode == "GC" else "ng-"
        for k in (50, 100):
            hits = [r[mode].get(str(k), r[mode].get(k)) for r in images]
            if any(type(h) is not int or not 0 <= h <= r["support"] for h, r in zip(hits, images, strict=True)):
                raise ValueError("Invalid per-image zero-shot hits")
            value = sum(h / r["support"] for h, r in zip(hits, images, strict=True)) / len(images)
            if not math.isclose(value, summary[f"{prefix}zR@{k}"], abs_tol=1e-12, rel_tol=0):
                raise ValueError("Zero-shot image-macro summary differs")
            if sum(hits) != summary["matched_zero_shot_annotations"][str(k)]:
                raise ValueError("Zero-shot total hits differ")


@torch.inference_mode()
def evaluate(model, dataset, device, batch_size, seen, cohort, old):
    zero = {"GC": [], "NG": []}
    full = {"GC": [], "NG": []}
    per_image = []
    index = 0
    for _, graphs in graph_batches(dataset, len(dataset), batch_size):
        batch, slices, skipped = collate_gnn_graphs(graphs)
        if skipped:
            raise ValueError("Evaluation skipped an image")
        logits = forward_gat_batch(model, batch, device).cpu()
        for graph, (start, end) in zip(graphs, slices, strict=True):
            triples = class_triplets(graph.gt_relation_index, graph.gt_predicate, graph.object_labels)
            expected = cohort["images"][index]
            support = sum(t not in seen for t in triples)
            if support != expected["zero_shot_annotations"] or len(triples) != expected["gt_annotations"]:
                raise ValueError("Graph GT disagrees with raw-HDF5 validation cohort")
            args = (
                logits[start:end],
                graph.edge_label_index,
                graph.gt_relation_index,
                graph.gt_predicate,
                graph.boxes,
                graph.object_labels,
            )
            item = {"validation_index": index, "image_id": str(graph.image_id), "support": support}
            for mode in ("GC", "NG"):
                constrained = mode == "GC"
                full[mode].append(evaluate_reference_image(*args, graph_constraint=constrained))
                z = evaluate_zero_shot_image(*args, seen, graph_constraint=constrained)
                zero[mode].append(z)
                item[mode] = z["hits"]
            if support:
                per_image.append(item)
            index += 1
    full_summaries = {}
    for mode, prefix in (("GC", ""), ("NG", "ng-")):
        summary = summarize_reference(full[mode], graph_constraint=mode == "GC")
        for k in (50, 100):
            for metric in ("R", "mR"):
                key = f"{prefix}{metric}@{k}"
                if not math.isclose(summary[key], old[mode][key], rel_tol=0, abs_tol=1e-12):
                    raise ValueError(f"Full-set metric differs from frozen earlier evaluation: {key}")
        if summary["per_predicate"] != old[mode]["per_predicate"]:
            raise ValueError("Full per-predicate results differ from earlier evaluation")
        full_summaries[mode] = {
            k: summary[k] for k in (f"{prefix}R@50", f"{prefix}R@100", f"{prefix}mR@50", f"{prefix}mR@100")
        }
    return {
        "zero_shot": {mode: summarize_zero_shot(rows, graph_constraint=mode == "GC") for mode, rows in zero.items()},
        "full_set_metric_reproduction": full_summaries,
        "zero_shot_images": per_image,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("data/raw/vg150"))
    p.add_argument("--output-dir", type=Path, default=Path("reports/level1/zero-shot-reference-v1"))
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--prepare-only", action="store_true")
    args = p.parse_args()
    if args.batch_size <= 0:
        p.error("batch-size must be positive")
    torch.set_num_threads(1)
    seen, vocabulary, cohort = prepare_cohort(args.root, args.output_dir)
    old_reports, summaries = {}, {}
    for mode, protocol, folder in (
        ("GC", GC_PROTOCOL, "reference-v1-fusion"),
        ("NG", NG_PROTOCOL, "reference-ng-v1-fusion"),
    ):
        path = Path(f"reports/level1/{folder}/15run-val5000-summary.json")
        summaries[mode], old_reports[mode] = checked_reports(path, protocol)
    entries = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            filename = f"{condition}-seed{seed}-val5000.json"
            gc, ng = (old_reports[m][filename] for m in ("GC", "NG"))
            compare_reports(gc, ng)
            if gc["data_sha256"] != vocabulary["data_sha256"]:
                raise ValueError("Train vocabulary inputs differ from frozen evaluations")
            entries.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "checkpoint": gc["checkpoint"],
                    "checkpoint_sha256": gc["checkpoint_sha256"],
                    "old_reports": {
                        mode: next(r for r in summaries[mode]["reports"] if Path(r["path"]).name == filename)
                        for mode in ("GC", "NG")
                    },
                }
            )
    write_fixed(
        args.output_dir / "frozen-checkpoints.json",
        encode({"entries": entries, "conditions": CONDITIONS, "seeds": SEEDS}),
    )
    print(
        f"Train unique triples: {len(seen)}; validation zero-shot: {cohort['zero_shot_annotations']} rows / {cohort['images_with_zero_shot_gt']} images / {cohort['unique_zero_shot_triples']} unique triples",
        flush=True,
    )
    if args.prepare_only:
        return
    if not cohort["images_with_zero_shot_gt"]:
        raise ValueError("No zero-shot GT in this validation cohort")
    device = resolve_device(args.device)
    definitions = {
        name: sha256(args.output_dir / name)
        for name in ("train-seen-triplets.json", "validation-cohort.json", "frozen-checkpoints.json")
    }
    code_hashes = source_hashes()
    dataset, config_ref, reports = None, None, []
    for i, entry in enumerate(entries, 1):
        path = args.output_dir / f"{entry['condition']}-seed{entry['seed']}-val5000.json"
        if sha256(Path(entry["checkpoint"])) != entry["checkpoint_sha256"]:
            raise ValueError("Frozen checkpoint changed")
        print(f"[{i}/9] {entry['condition']} seed {entry['seed']}", flush=True)
        if path.exists():
            report = json.loads(path.read_text())
        else:
            checkpoint = torch.load(entry["checkpoint"], map_location="cpu", weights_only=True)
            config = checkpoint["config"]
            filename = path.name
            if config != old_reports["GC"][filename]["config"] or config["model"] != "gat_kg_fusion":
                raise ValueError("Unexpected checkpoint config")
            if config["data_sha256"] != vocabulary["data_sha256"] or config["num_classes"] != 51:
                raise ValueError("Checkpoint uses different data or vocabulary")
            cache = Path(config["visual_cache"])
            if dataset is None:
                load_visual_manifest(cache, args.root, config["box_policy"])
                if sha256(cache / "manifest.json") != config["visual_cache_manifest_sha256"]:
                    raise ValueError("Changed visual cache")
                dataset = make_dataset(args.root, "val", config["box_policy"], cache)
                config_ref = config
            elif any(
                config[k] != config_ref[k] for k in ("visual_cache", "visual_cache_manifest_sha256", "box_policy")
            ):
                raise ValueError("Checkpoint inputs differ")
            if len(dataset) != 5000:
                raise ValueError("Expected full validation")
            model = load_model(checkpoint).to(device).eval()
            measured = evaluate(
                model,
                dataset,
                device,
                args.batch_size,
                seen,
                cohort,
                {m: old_reports[m][filename] for m in ("GC", "NG")},
            )
            report = {
                "protocol": PROTOCOL,
                "reference_commit": REFERENCE_COMMIT,
                "condition": entry["condition"],
                "seed": entry["seed"],
                "checkpoint": entry["checkpoint"],
                "checkpoint_sha256": entry["checkpoint_sha256"],
                "checkpoint_epoch": checkpoint["epoch"],
                "split": "val",
                "images_evaluated": len(dataset),
                "full_split": True,
                "definitions": definitions,
                "evaluator_sha256": code_hashes,
                "device": str(device),
                "batch_size": args.batch_size,
                "torch_version": torch.__version__,
                "numpy_version": np.__version__,
                "recall_aggregation": "image_macro_over_images_with_zero_shot_gt",
                "checkpoint_selection": "historical_legacy_mR50_not_reselected",
                **measured,
            }
            write_fixed(path, encode(report))
            del model, checkpoint
        verify_report(report, entry, definitions, code_hashes, cohort)
        reports.append(report)
    result = {
        "protocol": PROTOCOL,
        "split": "val",
        "images_evaluated": 5000,
        "seeds": list(SEEDS),
        "definitions": definitions,
        "metrics_unit": "fraction_0_to_1",
        "delta_unit": "percentage_points",
        "std_definition": "sample_standard_deviation_ddof_1",
        "groups": [],
        "paired_effects": [],
        "reports": [
            {
                "path": str(args.output_dir / f"{r['condition']}-seed{r['seed']}-val5000.json"),
                "sha256": sha256(args.output_dir / f"{r['condition']}-seed{r['seed']}-val5000.json"),
            }
            for r in reports
        ],
    }
    lookup = {(r["condition"], r["seed"]): r for r in reports}
    for condition in CONDITIONS:
        metrics = {
            key: stats([lookup[condition, s]["zero_shot"][mode][key] for s in SEEDS])
            for mode, prefix in (("GC", ""), ("NG", "ng-"))
            for key in (f"{prefix}zR@50", f"{prefix}zR@100")
        }
        result["groups"].append({"condition": condition, "metrics": metrics})
    for control in ("baseline", "pair_shuffled"):
        result["paired_effects"].append(
            {
                "treatment": "pair_real",
                "control": control,
                "metrics": {
                    key: stats(
                        [
                            100
                            * (
                                lookup["pair_real", s]["zero_shot"][mode][key]
                                - lookup[control, s]["zero_shot"][mode][key]
                            )
                            for s in SEEDS
                        ]
                    )
                    for mode, prefix in (("GC", ""), ("NG", "ng-"))
                    for key in (f"{prefix}zR@50", f"{prefix}zR@100")
                },
            }
        )
    write_fixed(args.output_dir / "9run-val5000-summary.json", encode(result))
    print(encode(result["groups"]))


if __name__ == "__main__":
    main()
