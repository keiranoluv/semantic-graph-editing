"""GQA supervision retaining multi-predicate pairs without overweighting duplicates."""

from collections import Counter

import torch
import torch.nn.functional as F


def soft_targets(graph, num_classes):
    """Uniform mass over distinct GT predicates; background only for unannotated pairs.

    This is soft-target cross entropy, not independent sigmoid/BCE classification.
    Duplicate rows are preserved in graph.gt_* for evaluation, but add no train weight.
    """
    pairs = graph.edge_label_index
    targets = torch.zeros((pairs.shape[1], num_classes), dtype=torch.float32)
    lookup = {tuple(pair): i for i, pair in enumerate(pairs.T.tolist())}
    if len(lookup) != pairs.shape[1]:
        raise ValueError("Duplicate candidate pairs")
    for pair, predicate in zip(graph.gt_relation_index.T.tolist(), graph.gt_predicate.tolist(), strict=True):
        if not 0 < predicate < num_classes or tuple(pair) not in lookup:
            raise ValueError("GT outside candidates/vocabulary")
        targets[lookup[tuple(pair)], predicate] = 1
    empty = targets.sum(1) == 0
    targets[empty, 0] = 1
    return targets / targets.sum(1, keepdim=True)


def sample_pairs(targets, generator, negative_ratio=3, empty_negatives=32):
    if negative_ratio < 0 or empty_negatives < 0:
        raise ValueError("Invalid negative sampling limits")
    positives = torch.where(targets[:, 0] == 0)[0]
    negatives = torch.where(targets[:, 0] == 1)[0]
    count = min(len(negatives), negative_ratio * len(positives) if len(positives) else empty_negatives)
    chosen = negatives[torch.randperm(len(negatives), generator=generator)[:count]]
    return torch.cat((positives, chosen))


def collate(graphs, num_classes, generator=None):
    """Sample supervision only; GAT always receives the full label-free topology."""
    xs, labels, edges, pairs, targets, slices = [], [], [], [], [], []
    node_offset, pair_offset = 0, 0
    for graph in graphs:
        target = soft_targets(graph, num_classes)
        selected = (
            sample_pairs(target, generator) if generator is not None else torch.arange(len(target), dtype=torch.long)
        )
        xs.append(graph.x)
        labels.append(graph.object_labels)
        edges.append(graph.edge_index + node_offset)
        pairs.append(graph.edge_label_index[:, selected] + node_offset)
        targets.append(target[selected])
        slices.append((pair_offset, pair_offset + len(selected)))
        node_offset += graph.num_nodes
        pair_offset += len(selected)
    if not graphs:
        raise ValueError("Cannot collate an empty batch")
    return {
        "x": torch.cat(xs),
        "object_labels": torch.cat(labels),
        "edge_index": torch.cat(edges, dim=1),
        "pairs": torch.cat(pairs, dim=1),
        "targets": torch.cat(targets),
    }, slices


def forward(model, kind, batch, device):
    kwargs = {
        "x": batch["x"].to(device),
        "object_labels": batch["object_labels"].to(device),
        "edge_label_index": batch["pairs"].to(device),
    }
    if kind == "gat":
        kwargs["edge_index"] = batch["edge_index"].to(device)
    return model(**kwargs)


def cross_entropy_sum(logits, targets):
    if not torch.isfinite(logits).all():
        raise RuntimeError("Non-finite logits")
    return -(targets * F.log_softmax(logits, dim=1)).sum()


def graph_statistics(graphs, num_classes):
    counts = Counter(images=len(graphs))
    predicate_support = Counter()
    for g in graphs:
        targets = soft_targets(g, num_classes)
        counts.update(
            objects=g.num_nodes,
            candidate_pairs=len(targets),
            positive_pairs=int((targets[:, 0] == 0).sum()),
            background_pairs=int((targets[:, 0] == 1).sum()),
            multi_predicate_pairs=int(((targets[:, 1:] > 0).sum(1) > 1).sum()),
            gt_annotation_rows=g.gt_predicate.numel(),
            unique_gt_triplets=len(set(zip(*g.gt_relation_index.tolist(), g.gt_predicate.tolist(), strict=True))),
            images_without_gt=int(g.gt_predicate.numel() == 0),
            images_without_candidates=int(len(targets) == 0),
        )
        predicate_support.update(g.gt_predicate.tolist())
    return {**dict(counts), "predicates_with_gt": len(predicate_support)}
