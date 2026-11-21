"""Versioned PredCls metrics matching KaihuaTang's pinned evaluator semantics.

Reference: Scene-Graph-Benchmark.pytorch @ ceb71fa88461c2a97a6258a80f47669d89207296.
Uses supplied GT boxes/classes, inclusive XYXY IoU, and retains duplicate GT rows.
"""

from collections import Counter

import numpy as np
import torch

PROTOCOL = "PredCls_reference_v1_iou50_image_macro_per_predicate_50"
NG_PROTOCOL = "PredCls_reference_ng_v1_iou50_image_macro_per_predicate_50"
REFERENCE_COMMIT = "ceb71fa88461c2a97a6258a80f47669d89207296"


def box_iou(boxes):
    """Pairwise IoU with the reference BoxList's inclusive pixel convention."""
    lo = torch.maximum(boxes[:, None, :2], boxes[None, :, :2])
    hi = torch.minimum(boxes[:, None, 2:], boxes[None, :, 2:])
    intersection = (hi - lo + 1).clamp(min=0).prod(-1)
    area = (boxes[:, 2:] - boxes[:, :2] + 1).prod(-1)
    return intersection / (area[:, None] + area[None, :] - intersection)


def evaluate_reference_image(
    logits, candidate_pairs, gt_pairs, gt_predicates, boxes, object_labels, ks=(50, 100), *, graph_constraint=True
):
    """Count matched GT annotation rows, allowing class/IoU instance matching."""
    if logits.ndim != 2 or logits.shape[1] < 2 or not torch.isfinite(logits).all():
        raise ValueError("Expected finite logits [pairs, background + predicates]")
    if candidate_pairs.shape != (2, logits.shape[0]) or gt_pairs.shape != (2, gt_predicates.numel()):
        raise ValueError("Invalid relation shapes")
    if boxes.ndim != 2 or boxes.shape[1] != 4 or object_labels.shape != (len(boxes),):
        raise ValueError("Invalid object boxes or labels")
    if not torch.isfinite(boxes).all() or (boxes[:, 2:] < boxes[:, :2]).any():
        raise ValueError("Invalid XYXY boxes")
    if not ks or any(k <= 0 for k in ks):
        raise ValueError("K must be positive")
    for indices in (candidate_pairs, gt_pairs):
        if indices.dtype != torch.long or (indices.numel() and (indices.min() < 0 or indices.max() >= len(boxes))):
            raise ValueError("Invalid object indices")
    if object_labels.dtype != torch.long or (object_labels <= 0).any():
        raise ValueError("Expected foreground object labels")
    if gt_predicates.dtype != torch.long or ((gt_predicates <= 0) | (gt_predicates >= logits.shape[1])).any():
        raise ValueError("Invalid GT predicates")
    # PredCls uses unit GT object scores. Match the reference torch.sort policy.
    probabilities = logits.softmax(-1)[:, 1:]
    scores, labels = probabilities.max(-1)
    labels = labels + 1
    pair_order = scores.sort(descending=True).indices
    if graph_constraint:
        ranking = pair_order[: max(ks)]
        selected = candidate_pairs[:, ranking]
        selected_labels = labels[ranking]
    else:
        # Upstream receives pairs in graph-constrained postprocessor order, then
        # argsort_desc ranks ALL foreground triplets with NumPy's default sort.
        # Preserve both steps, including ties; PredCls object scores are one.
        all_scores = probabilities[pair_order].detach().cpu().numpy().astype(np.float64)
        flat_order = np.argsort(-all_scores.ravel())[: max(ks)]
        pair_ids, predicate_ids = np.unravel_index(flat_order, all_scores.shape)
        selected = candidate_pairs[:, pair_order[torch.as_tensor(pair_ids, device=logits.device)]]
        selected_labels = torch.as_tensor(predicate_ids + 1, device=logits.device)
    gs, go = gt_pairs
    ps, po = selected
    overlap = box_iou(boxes.float())
    matches = (
        (object_labels[gs, None] == object_labels[ps][None, :])
        & (object_labels[go, None] == object_labels[po][None, :])
        & (gt_predicates[:, None] == selected_labels[None, :])
        & (overlap[gs[:, None], ps[None, :]] >= 0.5)
        & (overlap[go[:, None], po[None, :]] >= 0.5)
    )
    hits, class_hits = {}, {}
    for k in ks:
        found = matches[:, :k].any(dim=1)
        hits[k] = int(found.sum())
        class_hits[k] = Counter(gt_predicates[found].tolist())
    return {
        "support": gt_predicates.numel(),
        "hits": hits,
        "class_support": Counter(gt_predicates.tolist()),
        "class_hits": class_hits,
    }


def summarize_reference(results, num_predicates=50, ks=(50, 100), *, graph_constraint=True):
    """Average per-image class recalls, then over the fixed foreground vocabulary."""
    if num_predicates <= 0:
        raise ValueError("Expected a positive predicate count")
    valid = [r for r in results if r["support"] > 0]
    support, image_counts = Counter(), Counter()
    sums = {k: Counter() for k in ks}
    for r in valid:
        support.update(r["class_support"])
        for c, count in r["class_support"].items():
            if not 1 <= c <= num_predicates or count <= 0:
                raise ValueError("Predicate support outside configured vocabulary")
            image_counts[c] += 1
            for k in ks:
                sums[k][c] += r["class_hits"][k][c] / count
    prefix = "" if graph_constraint else "ng-"
    per_class = {
        str(c): {
            "support": support[c],
            "images_with_gt": image_counts[c],
            **{f"{prefix}R@{k}": sums[k][c] / image_counts[c] if image_counts[c] else 0.0 for k in ks},
        }
        for c in range(1, num_predicates + 1)
    }
    summary = {
        "num_images": len(results),
        "num_images_with_gt": len(valid),
        "num_predicates_with_gt": len(support),
        "num_predicates_for_mean": num_predicates,
        "per_predicate": per_class,
    }
    for k in ks:
        summary[f"{prefix}R@{k}"] = sum(r["hits"][k] / r["support"] for r in valid) / len(valid) if valid else None
        summary[f"{prefix}mR@{k}"] = (
            sum(row[f"{prefix}R@{k}"] for row in per_class.values()) / num_predicates if valid else None
        )
    return summary
