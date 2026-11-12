"""Read-only, snapshot-based full-validation diagnosis; never opens project test."""

import gc
import hashlib
import io
import json
import time
from collections import Counter
from pathlib import Path

import torch

from src.datasets.gqa_full import GQAFullDataset, atomic_write, digest, verify_cache
from src.evaluation.gqa_pilot import summarize_gqa
from src.evaluation.reference_predcls import evaluate_reference_image
from src.knowledge.transe.core import sha256
from src.training.gqa_full import create_model, load_vectors, setup_runtime
from src.training.gqa_pilot import collate, forward


def grouped_metrics(rows, summary, ids):
    ids = set(ids)
    per = summary["per_predicate"]
    supported = [c for c in ids if per[str(c)]["support"] > 0]
    output = {
        "classes": len(ids),
        "classes_with_gt": len(supported),
        "gt_rows": sum(per[str(c)]["support"] for c in ids),
    }
    for k in (50, 100):
        recalls = []
        for row in rows:
            support = sum(row["class_support"].get(c, 0) for c in ids)
            if support:
                recalls.append(sum(row["class_hits"][k].get(c, 0) for c in ids) / support)
        output[f"R@{k}"] = sum(recalls) / len(recalls) if recalls else None
        output[f"mR@{k}"] = sum(per[str(c)][f"R@{k}"] for c in ids) / len(ids) if ids else None
        output[f"mR_supported@{k}"] = (
            sum(per[str(c)][f"R@{k}"] for c in supported) / len(supported) if supported else None
        )
        output[f"classes_recalled@{k}"] = sum(per[str(c)][f"R@{k}"] > 0 for c in supported)
    return output


def diagnose_image(logits, graph, targets, excluded):
    args = (graph.edge_label_index, graph.gt_relation_index, graph.gt_predicate, graph.boxes, graph.object_labels)
    reference = evaluate_reference_image(logits, *args)
    keep = ~torch.isin(graph.gt_predicate, torch.tensor(sorted(excluded)))
    masked = logits.clone()
    masked[:, sorted(excluded)] = -1e9
    # Counterfactual: remove two labels, renormalize, re-rank and evaluate retained GT.
    counterfactual = evaluate_reference_image(
        masked,
        graph.edge_label_index,
        graph.gt_relation_index[:, keep],
        graph.gt_predicate[keep],
        graph.boxes,
        graph.object_labels,
    )
    probabilities = logits.softmax(-1)[:, 1:]
    scores, labels = probabilities.max(-1)
    labels = labels + 1
    ranking = scores.sort(descending=True).indices[:50]
    predictions = Counter(labels[ranking].tolist())
    positive = torch.where(targets[:, 0] == 0)[0]
    allowed = [i for i in range(1, logits.shape[1]) if i not in excluded]
    non_spatial = torch.where((targets[:, allowed] > 0).any(1))[0]
    full_labels = logits.argmax(1)
    stats = Counter(
        all_pairs=len(logits),
        background_predictions=int((full_labels == 0).sum()),
        gt_unique_pairs=len(positive),
        gt_pairs_correct_foreground=int((targets[positive, labels[positive]] > 0).sum()),
        gt_pairs_correct_with_background=int((targets[positive, full_labels[positive]] > 0).sum()),
        non_spatial_gt_unique_pairs=len(non_spatial),
        non_spatial_gt_pairs_correct_foreground=int(
            (
                (targets[non_spatial, labels[non_spatial]] > 0)
                & ~torch.isin(labels[non_spatial], torch.tensor(sorted(excluded)))
            ).sum()
        ),
    )
    return reference, counterfactual, predictions, stats


def main():
    output = Path("reports/level2/gqa-quality-v1")
    snapshots = Path("runs/diagnostics/gqa-quality-v1")
    if output.exists() or snapshots.exists():
        raise FileExistsError("Use a fresh version; diagnostic evidence is immutable")
    root = Path("runs/level2/gqa-full-v3-bs512")
    experiment = json.loads((root / "experiment.json").read_text())
    config, dimensions = experiment["config"], experiment["dimensions"]
    assert all(sha256(Path(k)) == v for k, v in experiment["code_sha256"].items())
    manifest = json.loads(Path(config["manifest"]).read_text())
    assert sha256(Path(config["manifest"])) == experiment["manifest_sha256"]
    vocabulary = manifest["vocabulary"]["predicate_to_idx"]
    excluded = {vocabulary["to the left of"], vocabulary["to the right of"]}
    audit_path = Path("reports/level2/gqa-scale-audit-v1.json")
    audit = json.loads(audit_path.read_text())
    assert audit["manifest_sha256"] == experiment["manifest_sha256"]
    counts = {r["id"]: r["annotation_rows"]["train_full"] for r in audit["per_predicate"]}
    ordered = sorted(counts, key=lambda c: (-counts[c], c))
    groups = {
        "all": ordered,
        "left_right": sorted(excluded),
        "non_spatial": [c for c in ordered if c not in excluded],
        "train_top10": ordered[:10],
        "train_next40": ordered[10:50],
        "train_remaining260": ordered[50:],
    }
    snapshots.mkdir(parents=True)
    output.mkdir(parents=True)
    selected = []
    for kind in config["models"]:
        for seed in config["seeds"]:
            directory = root / kind / str(seed)
            source = directory / "best.pt"
            if not source.exists():
                continue
            raw = source.read_bytes()  # One immutable inode even if training replaces best.pt.
            checkpoint = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
            expected_identity = {"experiment_sha256": digest(experiment), "model": kind, "seed": seed}
            assert checkpoint["identity"] == expected_identity
            target = snapshots / f"{kind}-{seed}.pt"
            target.write_bytes(raw)
            selected.append(
                {
                    "model": kind,
                    "seed": seed,
                    "snapshot": str(target),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "selected_epoch": checkpoint["epoch"],
                    "stored_mR@50": checkpoint["score"],
                    "training_complete_at_snapshot": (directory / "result.json").exists(),
                }
            )
    metadata = verify_cache(config["cache"])
    assert sha256(Path(config["cache"]) / "manifest.json") == experiment["cache_manifest_sha256"]
    setup_runtime(config)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(3 * 1024**3 / total)
    vectors = load_vectors(config, dimensions)
    dataset = GQAFullDataset(config["cache"], metadata, "val")
    try:
        graphs = dataset[:]
    finally:
        dataset.close()
    plan = {
        "scope": "full 5000-image validation diagnosis, not a test benchmark",
        "selected": selected,
        "batch_size": 64,
        "cuda_allocator_limit_bytes": 3 * 1024**3,
        "group_class_ids": groups,
        "source_sha256": {str(Path(__file__).relative_to(Path.cwd())): sha256(Path(__file__))},
        "audit_sha256": sha256(audit_path),
        "experiment_sha256": digest(experiment),
        "validation_image_ids_sha256": digest([g.image_id for g in graphs]),
        "test_used": False,
        "counterfactual": "mask left/right logits to -1e9, renormalize and rerank; discard left/right GT; diagnostic only",
        "partial_runs": "unfinished checkpoints are exploratory, not matched converged comparisons",
    }
    atomic_write(output / "manifest.json", plan)
    for item in selected:
        started = time.perf_counter()
        checkpoint = torch.load(item["snapshot"], map_location="cpu", weights_only=True)
        model = create_model(item["model"], dimensions, vectors).cuda()
        model.load_state_dict(checkpoint["model"])
        model.eval()
        rows, masked_rows, predictions, pairs = [], [], Counter(), Counter()
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            for begin in range(0, len(graphs), 64):
                group = graphs[begin : begin + 64]
                batch, slices = collate(group, dimensions["num_classes"])
                logits = forward(
                    model, "penet" if item["model"].startswith("pe-") else item["model"], batch, "cuda"
                ).cpu()
                for graph, (a, b) in zip(group, slices, strict=True):
                    row, masked, prediction, stats = diagnose_image(logits[a:b], graph, batch["targets"][a:b], excluded)
                    rows.append(row)
                    masked_rows.append(masked)
                    predictions.update(prediction)
                    pairs.update(stats)
                if begin % 1024 == 0:
                    print(
                        f"[DIAG] {item['model']}/{item['seed']} images={min(begin+64,len(graphs))}/{len(graphs)}",
                        flush=True,
                    )
        summary = summarize_gqa(rows, dimensions["num_classes"] - 1)
        masked_summary = summarize_gqa(masked_rows, dimensions["num_classes"] - 1)
        assert all(
            summary["per_predicate"][str(r["id"])]["support"] == r["annotation_rows"]["val_full"]
            for r in audit["per_predicate"]
        )
        delta = abs(summary["mR@50"] - item["stored_mR@50"])
        report = {
            **item,
            "seconds": time.perf_counter() - started,
            "gc": summary,
            "groups": {name: grouped_metrics(rows, summary, ids) for name, ids in groups.items()},
            "counterfactual_non_spatial": grouped_metrics(masked_rows, masked_summary, groups["non_spatial"]),
            "top50_predictions_by_class": dict(predictions),
            "pair_counts": dict(pairs),
            "top50_left_right_fraction": sum(predictions[c] for c in excluded) / sum(predictions.values()),
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "stored_mR_absolute_difference": delta,
            "batch_size_note": "inference uses 64 instead of training validation 512; numerical agreement checked",
            "test_used": False,
        }
        atomic_write(output / f"{item['model']}-{item['seed']}.json", report)
        print(
            "[RESULT]",
            json.dumps(
                {
                    k: report[k]
                    for k in [
                        "model",
                        "seed",
                        "selected_epoch",
                        "stored_mR_absolute_difference",
                        "top50_left_right_fraction",
                        "peak_cuda_allocated_bytes",
                    ]
                }
            ),
            flush=True,
        )
        del model, checkpoint, batch, logits
        gc.collect()
        torch.cuda.empty_cache()
    print("[DONE]", str(output), flush=True)


if __name__ == "__main__":
    main()
