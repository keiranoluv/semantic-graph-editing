"""Zero-shot PredCls: unseen class triples, with unchanged whole-image prediction ranking."""

from collections import Counter

import torch

from src.evaluation.reference_predcls import evaluate_reference_image

PROTOCOL = "PredCls_zero_shot_reference_v1_train_raw_class_triples_image_macro"


def class_triplets(gt_pairs, predicates, object_labels):
    """Use explicit (subject class, predicate, object class) order, never node IDs."""
    return list(
        zip(object_labels[gt_pairs[0]].tolist(), predicates.tolist(), object_labels[gt_pairs[1]].tolist(), strict=True)
    )


def unseen_mask(gt_pairs, predicates, object_labels, train_seen):
    return torch.tensor(
        [triple not in train_seen for triple in class_triplets(gt_pairs, predicates, object_labels)],
        dtype=torch.bool,
        device=predicates.device,
    )


def evaluate_zero_shot_image(
    logits,
    candidate_pairs,
    gt_pairs,
    predicates,
    boxes,
    object_labels,
    train_seen,
    ks=(50, 100),
    *,
    graph_constraint=True,
):
    mask = unseen_mask(gt_pairs, predicates, object_labels, train_seen)
    # Filter only the GT being scored. ALL pairs/predicate scores remain eligible
    # for top-K, so seen predictions can consume slots before unseen GT matches.
    return evaluate_reference_image(
        logits,
        candidate_pairs,
        gt_pairs[:, mask],
        predicates[mask],
        boxes,
        object_labels,
        ks=ks,
        graph_constraint=graph_constraint,
    )


def summarize_zero_shot(results, ks=(50, 100), *, graph_constraint=True):
    valid = [row for row in results if row["support"] > 0]
    support = Counter()
    for row in valid:
        support.update(row["class_support"])
    prefix = "" if graph_constraint else "ng-"
    return {
        "images_evaluated": len(results),
        "images_with_zero_shot_gt": len(valid),
        "zero_shot_gt_annotations": sum(row["support"] for row in valid),
        "zero_shot_gt_by_predicate": dict(sorted(support.items())),
        "matched_zero_shot_annotations": {str(k): sum(row["hits"][k] for row in valid) for k in ks},
        **{
            f"{prefix}zR@{k}": sum(row["hits"][k] / row["support"] for row in valid) / len(valid) if valid else None
            for k in ks
        },
    }
