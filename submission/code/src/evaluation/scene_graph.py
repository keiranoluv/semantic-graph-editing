"""Graph-constrained PredCls Recall@K evaluation."""

from collections import Counter

import torch


def evaluate_image(
    logits: torch.Tensor,
    candidate_pairs: torch.Tensor,
    gt_pairs: torch.Tensor,
    gt_predicates: torch.Tensor,
    ks: tuple[int, ...] = (50, 100),
) -> dict:
    """Evaluate a single image using top-1 predicate per ordered pair."""
    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError("Expected logits [E, background + predicates]")

    if candidate_pairs.shape != (2, logits.shape[0]):
        raise ValueError("Candidate pair shape mismatch")

    if gt_pairs.shape != (2, gt_predicates.numel()):
        raise ValueError("Ground-truth shape mismatch")

    if any(k <= 0 for k in ks):
        raise ValueError("K must be positive")

    if not torch.isfinite(logits).all():
        raise ValueError("Non-finite logits")

    # Use the full softmax, including the background class.
    scores = logits.softmax(dim=-1)[:, 1:]
    best_scores, best_classes = scores.max(dim=-1)
    best_classes = best_classes + 1

    predictions = [
        (int(candidate_pairs[0, i]), int(best_classes[i]), int(candidate_pairs[1, i])) for i in range(logits.shape[0])
    ]

    ranked = torch.argsort(best_scores, descending=True, stable=True).tolist()

    # Identical annotated triplets are counted once.
    # Different predicates for the same pair are retained.
    ground_truth = {
        (int(gt_pairs[0, i]), int(gt_predicates[i]), int(gt_pairs[1, i])) for i in range(gt_predicates.numel())
    }

    if any(p <= 0 or p >= logits.shape[1] for _, p, _ in ground_truth):
        raise ValueError("Ground-truth predicate outside class range")

    hits = {}
    class_hits = {}
    class_support = Counter(p for _, p, _ in ground_truth)

    for k in ks:
        selected = {predictions[i] for i in ranked[:k]}
        matched = selected & ground_truth

        hits[k] = len(matched)
        class_hits[k] = Counter(p for _, p, _ in matched)

    return {
        "support": len(ground_truth),
        "hits": hits,
        "class_support": class_support,
        "class_hits": class_hits,
    }


def summarize_evaluation(results: list[dict], ks=(50, 100)) -> dict:
    """Aggregate image-macro Recall and predicate-macro mean Recall."""
    output = {"num_images": len(results)}
    valid = [item for item in results if item["support"] > 0]

    output["num_images_with_gt"] = len(valid)

    support = Counter()
    for item in valid:
        support.update(item["class_support"])

    output["num_predicates_with_gt"] = len(support)

    for k in ks:
        if not valid:
            output[f"R@{k}"] = None
            output[f"mR@{k}"] = None
            continue

        output[f"R@{k}"] = sum(item["hits"][k] / item["support"] for item in valid) / len(valid)

        class_hits = Counter()
        for item in valid:
            class_hits.update(item["class_hits"][k])

        output[f"mR@{k}"] = sum(class_hits[c] / support[c] for c in support) / len(support)

    return output
