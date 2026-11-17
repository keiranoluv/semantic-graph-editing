"""CPU long-tail for 54 test reports; nine frozen fusion zero-shot runs, no training."""

import argparse
import fcntl
import json
import math
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Subset

from scripts.evaluate_level1 import load_model
from scripts.train_mlp import make_dataset
from src.datasets.batching import graph_batches
from src.evaluation.zero_shot import PROTOCOL, class_triplets, evaluate_zero_shot_image
from src.features.visual import load_visual_manifest
from src.graph.gnn_batching import collate_gnn_graphs, forward_gat_batch
from tools.analyze_level1_long_tail import DATA_FILES, SEEDS, encode, grouped_recall, stats, write_fixed
from tools.level1_loss_fusion_test27_v1 import build_plan
from tools.level1_reference_test27_v1 import PLAN, PLAN_HASH, ROOT, sha256, validate
from tools.level1_zero_shot import raw_class_triples, seen_from_rows

OUTPUT = ROOT / "reports/level1/test-tail-zero-v1"
DATA = ROOT / "data/raw/vg150"
GROUPS = ROOT / "reports/level1/long-tail-reference-v1/predicate-groups.json"
VOCAB = ROOT / "reports/level1/zero-shot-reference-v1/train-seen-triplets.json"
CONDITIONS = ("fusion_baseline", "fusion_pair_real", "fusion_pair_shuffled")


def source_hashes():
    paths = list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").glob("*.py"))
    paths += [
        ROOT / "tools" / name
        for name in (
            "level1_test_tail_zero_v1.py",
            "analyze_level1_long_tail.py",
            "level1_zero_shot.py",
            "level1_reference_test27_v1.py",
            "level1_loss_fusion_test27_v1.py",
        )
    ]
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(set(paths))}


def checked_test_reports():
    if sha256(PLAN) != PLAN_HASH:
        raise ValueError("Original plan changed")
    result = []
    for folder, plan in (
        ("reference-test27-v1", json.loads(PLAN.read_text())),
        ("reference-loss-fusion-test27-v1", build_plan()),
    ):
        directory = ROOT / "reports/level1" / folder
        summary = json.loads((directory / "summary.json").read_text())
        saved = {r["path"]: r["sha256"] for r in summary["reports"]}
        if len(saved) != 27:
            raise ValueError("Incomplete test summary")
        for entry in plan["entries"]:
            path = directory / f"{entry['condition']}-seed{entry['seed']}-test26446.json"
            row = json.loads(path.read_text())
            validate(row, entry, plan)
            if sha256(ROOT / entry["checkpoint"]) != entry["checkpoint_sha256"]:
                raise ValueError("Checkpoint changed")
            if saved[str(path.relative_to(ROOT))] != sha256(path):
                raise ValueError("Test report changed")
            result.append((entry, row, path))
    if len(result) != 54 or len({e["checkpoint_sha256"] for e, _, _ in result}) != 54:
        raise ValueError("Expected 54 unique checkpoints")
    return result


def long_tail(reports, output):
    definition = json.loads(GROUPS.read_text())
    groups, rows, lookup = definition["groups"], [], {}
    support = None
    for entry, report, path in reports:
        if report["data_sha256"] != definition["data_sha256"]:
            raise ValueError("Group definition and test data differ")
        measured = grouped_recall(report, groups, "")
        current = {g: sum(report["per_predicate"][str(c)]["support"] for c in ids) for g, ids in groups.items()}
        if support is not None and support != current:
            raise ValueError("Test support differs between models")
        support = current
        rows.append(
            {
                "condition": entry["condition"],
                "seed": entry["seed"],
                "metrics": measured,
                "report": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
            }
        )
        lookup.setdefault(entry["condition"], {})[entry["seed"]] = measured
    conditions = {}
    for condition, by_seed in lookup.items():
        if sorted(by_seed) != list(SEEDS):
            raise ValueError("Incomplete seed group")
        conditions[condition] = {
            g: {f"mR@{k}": stats([by_seed[s][g][f"mR@{k}"] for s in SEEDS]) for k in (50, 100)} for g in groups
        }
    result = {
        "split": "test",
        "images": 26446,
        "protocol": "GC reference-v1",
        "inference_performed": False,
        "group_definition_sha256": sha256(GROUPS),
        "test_annotation_support": support,
        "unit": "fraction_0_to_1",
        "conditions": conditions,
        "runs": rows,
    }
    write_fixed(output / "long-tail.json", encode(result))
    table = [
        "# Long-tail VG150 test — 54 checkpoint, CPU-only",
        "",
        "GC reference-v1; 26.446 ảnh. Nhóm 10/20/20 lớp giữ nguyên từ train.",
        "Mean ± sample std (%), seed 42/123/3407. Top-K toàn ảnh không đổi.",
        "",
        "| Cấu hình | Head mR@50 | Body mR@50 | Tail mR@50 | Head mR@100 | Body mR@100 | Tail mR@100 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for c, values in conditions.items():
        cells = [values[g][f"mR@{k}"] for k in (50, 100) for g in ("head", "body", "tail")]
        table.append(
            "| " + c + " | " + " | ".join(f"{x['mean']*100:.2f} ± {x['sample_std']*100:.2f}" for x in cells) + " |"
        )
    write_fixed(output / "long-tail.md", "\n".join(table) + "\n")


def prepare(output):
    reports = checked_test_reports()
    long_tail(reports, output)
    hashes = {name: sha256(DATA / name) for name in DATA_FILES}
    vocabulary = json.loads(VOCAB.read_text())
    if hashes != vocabulary["data_sha256"]:
        raise ValueError("Historical train vocabulary inputs changed")
    splits = json.loads((DATA / "splits.json").read_text())
    if [len(splits[s]) for s in ("train", "val", "test")] != [57723, 5000, 26446]:
        raise ValueError("Wrong split sizes")
    if any(set(splits[a]) & set(splits[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Split overlap")
    with h5py.File(DATA / "VG-SGG-with-attri.h5", "r") as h5:
        seen, count = seen_from_rows(h5, splits["train"])
        if seen != {tuple(t) for t in vocabulary["triples"]} or count != vocabulary["train_annotations"]:
            raise ValueError("Train-only vocabulary reconstruction differs")
        images, predicate_support, triples_support = [], Counter(), Counter()
        all_annotations = 0
        for i, triples in enumerate(raw_class_triples(h5, splits["test"])):
            unseen = [t for t in triples if t not in seen]
            all_annotations += len(triples)
            if unseen:
                images.append([i, splits["test"][i], len(triples), len(unseen)])
                predicate_support.update(t[1] for t in unseen)
                triples_support.update(unseen)
    cohort = {
        "split": "test",
        "split_images": 26446,
        "gt_annotations": all_annotations,
        "zero_shot_annotations": sum(triples_support.values()),
        "unique_zero_shot_triples": len(triples_support),
        "images_with_zero_shot_gt": len(images),
        "zero_shot_gt_by_predicate": dict(sorted(predicate_support.items())),
        "image_columns": ["test_index", "hdf5_row", "all_gt_annotations", "zero_shot_annotations"],
        "images": images,
        "vocabulary_path": str(VOCAB.relative_to(ROOT)),
        "vocabulary_sha256": sha256(VOCAB),
        "data_sha256": hashes,
    }
    if not images:
        raise ValueError("Empty zero-shot cohort")
    # Compact arrays keep the fixed cohort small without dropping image provenance.
    write_fixed(output / "test-cohort.json", json.dumps(cohort, ensure_ascii=False, separators=(",", ":")) + "\n")
    entries = [
        {**e, "old_report": str(p.relative_to(ROOT)), "old_report_sha256": sha256(p)}
        for e, _, p in reports
        if e["condition"] in CONDITIONS
    ]
    entries.sort(key=lambda e: (CONDITIONS.index(e["condition"]), e["seed"]))
    if len(entries) != 9:
        raise ValueError("Expected nine fusion checkpoints")
    freeze = {
        "entries": entries,
        "cohort_sha256": sha256(output / "test-cohort.json"),
        "source_sha256": source_hashes(),
    }
    write_fixed(output / "freeze.json", encode(freeze))
    print(
        f"CPU audit PASS: 54 long-tail reports; zero-shot {len(images)} images / {sum(triples_support.values())} GT",
        flush=True,
    )


def summarize_images(images):
    """Recompute image-macro recall from compact per-image hits; reject corrupt output."""
    if not images or len({r[0] for r in images}) != len(images):
        raise ValueError("Empty or duplicate image rows")
    for row in images:
        if len(row) != 6 or any(type(x) is not int for x in row) or row[1] <= 0:
            raise ValueError("Invalid image row")
        if any(not 0 <= h <= row[1] for h in row[2:]) or row[2] > row[3] or row[4] > row[5]:
            raise ValueError("Invalid or nonmonotonic hits")
    return {
        key: sum(row[column] / row[1] for row in images) / len(images)
        for column, key in enumerate(("zR@50", "zR@100", "ng-zR@50", "ng-zR@100"), 2)
    }


def verify_report(report, entry, freeze, cohort):
    expected = {
        "protocol": PROTOCOL,
        "split": "test",
        "condition": entry["condition"],
        "seed": entry["seed"],
        "checkpoint_sha256": entry["checkpoint_sha256"],
        "cohort_sha256": freeze["cohort_sha256"],
        "source_sha256": freeze["source_sha256"],
        "split_images": 26446,
        "images_inferred": len(cohort["images"]),
    }
    if any(report.get(k) != v for k, v in expected.items()):
        raise ValueError("Incompatible existing report")
    rows = report["images"]
    if [(r[0], r[1]) for r in rows] != [(r[0], r[3]) for r in cohort["images"]]:
        raise ValueError("Cohort image identity/support differs")
    actual = summarize_images(rows)
    if any(not math.isclose(actual[k], report["metrics"][k], abs_tol=1e-12, rel_tol=0) for k in actual):
        raise ValueError("Metrics differ from per-image hits")


@torch.inference_mode()
def infer(model, dataset, images, seen, device, batch_size):
    result = []
    subset = Subset(dataset, [r[0] for r in images])
    for begin, graphs in graph_batches(subset, len(subset), batch_size):
        batch, slices, skipped = collate_gnn_graphs(graphs)
        if skipped:
            raise ValueError("Skipped image")
        logits = forward_gat_batch(model, batch, device).cpu()
        for j, (graph, (start, end)) in enumerate(zip(graphs, slices, strict=True)):
            expected = images[begin + j]
            triples = class_triplets(graph.gt_relation_index, graph.gt_predicate, graph.object_labels)
            if len(triples) != expected[2] or sum(t not in seen for t in triples) != expected[3]:
                raise ValueError("Dataset GT differs from raw test cohort")
            args = (
                logits[start:end],
                graph.edge_label_index,
                graph.gt_relation_index,
                graph.gt_predicate,
                graph.boxes,
                graph.object_labels,
                seen,
            )
            gc = evaluate_zero_shot_image(*args)
            ng = evaluate_zero_shot_image(*args, graph_constraint=False)
            result.append([expected[0], expected[3], gc["hits"][50], gc["hits"][100], ng["hits"][50], ng["hits"][100]])
        print(f"  {len(result)}/{len(images)} zero-shot images", flush=True)
    return result


def load_inputs(entry):
    checkpoint = torch.load(ROOT / entry["checkpoint"], map_location="cpu", weights_only=True)
    old = json.loads((ROOT / entry["old_report"]).read_text())
    config = checkpoint["config"]
    if config != old["config"] or config["model"] != "gat_kg_fusion" or config["num_classes"] != 51:
        raise ValueError("Checkpoint config differs")
    cache = Path(config["visual_cache"])
    load_visual_manifest(cache, DATA, config["box_policy"])
    if sha256(cache / "manifest.json") != config["visual_cache_manifest_sha256"]:
        raise ValueError("Visual cache changed")
    dataset = make_dataset(DATA, "test", config["box_policy"], cache)
    if len(dataset) != 26446:
        raise ValueError("Wrong test dataset")
    return checkpoint, dataset


def run(output, device, batch_size, smoke=False, summarize_only=False):
    freeze = json.loads((output / "freeze.json").read_text())
    cohort = json.loads((output / "test-cohort.json").read_text())
    if source_hashes() != freeze["source_sha256"] or sha256(output / "test-cohort.json") != freeze["cohort_sha256"]:
        raise ValueError("Frozen source or cohort changed")
    if sha256(VOCAB) != cohort["vocabulary_sha256"] or any(
        sha256(DATA / k) != v for k, v in cohort["data_sha256"].items()
    ):
        raise ValueError("Vocabulary or dataset changed")
    seen = {tuple(t) for t in json.loads(VOCAB.read_text())["triples"]}
    existing = {}
    # Validate every checkpoint/output before any GPU call; valid reports are reused.
    for entry in freeze["entries"]:
        if (
            sha256(ROOT / entry["checkpoint"]) != entry["checkpoint_sha256"]
            or sha256(ROOT / entry["old_report"]) != entry["old_report_sha256"]
        ):
            raise ValueError("Frozen checkpoint/test report changed")
        path = output / f"{entry['condition']}-seed{entry['seed']}.json"
        if path.exists():
            row = json.loads(path.read_text())
            verify_report(row, entry, freeze, cohort)
            existing[path.name] = row
        elif summarize_only:
            raise ValueError("Missing report; summarize never launches inference")
    if smoke:
        checkpoint, dataset = load_inputs(freeze["entries"][0])
        model = load_model(checkpoint).eval()
        one = infer(model, dataset, cohort["images"][:2], seen, "cpu", 1)
        two = infer(model, dataset, cohort["images"][:2], seen, "cpu", 2)
        if one != two:
            raise ValueError("CPU batch invariance failed")
        print("CPU smoke PASS: real checkpoint, two test images, batch 1 vs 2")
        return
    rows = []
    for entry in freeze["entries"]:
        path = output / f"{entry['condition']}-seed{entry['seed']}.json"
        if path.name in existing:
            print(f"SKIP verified {path.name}", flush=True)
            row = existing[path.name]
        else:
            print(f"RUN {path.name} device={device} batch={batch_size}", flush=True)
            checkpoint, dataset = load_inputs(entry)
            model = load_model(checkpoint).to(device).eval()
            hits = infer(model, dataset, cohort["images"], seen, device, batch_size)
            row = {
                "protocol": PROTOCOL,
                "split": "test",
                "split_images": 26446,
                "images_inferred": len(hits),
                "condition": entry["condition"],
                "seed": entry["seed"],
                "checkpoint_sha256": entry["checkpoint_sha256"],
                "cohort_sha256": freeze["cohort_sha256"],
                "source_sha256": freeze["source_sha256"],
                "device": device,
                "batch_size": batch_size,
                "torch_version": str(torch.__version__),
                "numpy_version": np.__version__,
                "tf32": False,
                "selection": "fixed_legacy_validation_checkpoint_no_reselection",
                "image_columns": ["test_index", "support", "GC_hits50", "GC_hits100", "NG_hits50", "NG_hits100"],
                "images": hits,
                "metrics": summarize_images(hits),
            }
            verify_report(row, entry, freeze, cohort)
            write_fixed(path, json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            del model, checkpoint, dataset
            if device == "cuda":
                torch.cuda.empty_cache()
        rows.append(row)
    summary = {
        "split": "test",
        "split_images": 26446,
        "images_with_zero_shot_gt": len(cohort["images"]),
        "zero_shot_annotations": cohort["zero_shot_annotations"],
        "unit": "fraction_0_to_1",
        "groups": {
            c: {k: stats([r["metrics"][k] for r in rows if r["condition"] == c]) for k in rows[0]["metrics"]}
            for c in CONDITIONS
        },
        "reports": {
            f"{r['condition']}-seed{r['seed']}.json": sha256(output / f"{r['condition']}-seed{r['seed']}.json")
            for r in rows
        },
    }
    write_fixed(output / "zero-shot-summary.json", encode(summary))
    table = [
        "# Zero-shot VG150 test — 9 checkpoint fusion",
        "",
        f"GT zero-shot: {cohort['zero_shot_annotations']} annotation trên {len(cohort['images'])}/26446 ảnh.",
        "Chỉ inference ảnh có GT zero-shot; ranking toàn ảnh giữ nguyên, không lọc prediction.",
        "Mean ± sample std (%), seed 42/123/3407; chưa gặp được định nghĩa từ annotation train.",
        "",
        "| Cấu hình | GC zR@50 | GC zR@100 | NG zR@50 | NG zR@100 |",
        "|---|---:|---:|---:|---:|",
    ]
    for c, values in summary["groups"].items():
        table.append(
            "| "
            + c
            + " | "
            + " | ".join(f"{x['mean']*100:.2f} ± {x['sample_std']*100:.2f}" for x in values.values())
            + " |"
        )
    write_fixed(output / "zero-shot.md", "\n".join(table) + "\n")
    print("\n".join(table), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "smoke", "evaluate", "summarize"), required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("batch-size must be positive")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / ".launcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage == "prepare":
            prepare(args.output_dir)
        else:
            run(
                args.output_dir,
                args.device,
                args.batch_size,
                smoke=args.stage == "smoke",
                summarize_only=args.stage == "summarize",
            )


if __name__ == "__main__":
    main()
