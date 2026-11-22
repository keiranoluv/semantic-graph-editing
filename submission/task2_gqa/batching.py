"""Validate immutable CPU graphs once; prepare label-free buckets and query maps."""

from dataclasses import dataclass

import torch
from task2_gqa.models.aligng import validate_pairs
from task2_gqa.training.supervision import soft_targets


@dataclass(frozen=True)
class PreparedGraph:
    graph: object
    num_classes: int
    context_lookup: torch.Tensor
    positive: torch.Tensor
    negative: torch.Tensor
    positive_rows: torch.Tensor

    def __getattr__(self, key):
        return getattr(self.graph, key)


def prepare_graph(graph, num_classes=311, num_objects=1700, feature_dim=517):
    """Owned dataset graphs are immutable after preparation; do not cache caller mutations."""
    for value in (
        graph.x,
        graph.object_labels,
        graph.edge_index,
        graph.edge_label_index,
        graph.gt_relation_index,
        graph.gt_predicate,
    ):
        if value.device.type != "cpu":
            raise ValueError("Prepare graph validation on CPU, before transferring batches")
    if graph.x.shape != (graph.num_nodes, feature_dim) or not torch.isfinite(graph.x).all():
        raise ValueError("Invalid frozen features")
    if graph.object_labels.shape != (graph.num_nodes,) or graph.object_labels.dtype != torch.long:
        raise ValueError("Invalid object labels")
    if ((graph.object_labels < 0) | (graph.object_labels >= num_objects)).any():
        raise ValueError("Object vocabulary mismatch")
    for pairs in (graph.edge_index, graph.edge_label_index):
        validate_pairs(pairs, graph.num_nodes, full=True)
    context_keys, order = (graph.edge_index[0] * graph.num_nodes + graph.edge_index[1]).sort()
    candidates = graph.edge_label_index[0] * graph.num_nodes + graph.edge_label_index[1]
    lookup = order[torch.searchsorted(context_keys, candidates)]
    targets = soft_targets(graph, num_classes)
    positive = torch.where(targets[:, 0] == 0)[0]
    negative = torch.where(targets[:, 0] == 1)[0]
    return PreparedGraph(graph, num_classes, lookup, positive, negative, targets[positive])


@dataclass(frozen=True)
class Group:
    context_index: torch.Tensor
    context_mask: torch.Tensor
    query_index: torch.Tensor
    query_rows: torch.Tensor
    query_slots: torch.Tensor

    def to(self, device):
        return Group(*(getattr(self, name).to(device) for name in self.__dataclass_fields__))


@dataclass(frozen=True)
class ModelBatch:
    """Trusted factory output; no target/predicate annotation enters the model."""

    x: torch.Tensor
    object_labels: torch.Tensor
    context_pairs: torch.Tensor
    query_pairs: torch.Tensor
    query_context: torch.Tensor
    groups: tuple


def groups_for(context_ptr, query_ptr, num_classes, *, max_images=32, attention_elements=16_000_000, heads=2):
    """Bucket ONLY on context length. Padding masks exclude all non-context keys.

    Group membership/order never depends on labels or sampled query lengths.
    Empty-context images have no query pairs and are omitted from softmax.
    """
    buckets = {}
    for i, (a, b) in enumerate(zip(context_ptr[:-1], context_ptr[1:], strict=True)):
        count = b - a
        if count:
            buckets.setdefault((count - 1).bit_length(), []).append(i)
    groups = []
    for bucket in sorted(buckets):
        size = max(1, min(max_images, attention_elements // (heads * num_classes * 2**bucket)))
        images = buckets[bucket]
        for start in range(0, len(images), size):
            ids = images[start : start + size]
            ccounts = torch.tensor([context_ptr[i + 1] - context_ptr[i] for i in ids])
            qcounts = torch.tensor([query_ptr[i + 1] - query_ptr[i] for i in ids])
            cmask = torch.arange(int(ccounts.max()))[None, :] < ccounts[:, None]
            cindex = torch.tensor([context_ptr[i] for i in ids])[:, None] + torch.arange(cmask.shape[1])
            cindex = cindex.masked_fill(~cmask, 0)
            qmask = torch.arange(int(qcounts.max()))[None, :] < qcounts[:, None]
            qindex = torch.tensor([query_ptr[i] for i in ids])[:, None] + torch.arange(qmask.shape[1])
            rows = qindex[qmask]
            qindex = qindex.masked_fill(~qmask, 0)
            groups.append(Group(cindex, cmask, qindex, rows, torch.where(qmask.flatten())[0]))
    return tuple(groups)


def collate(graphs, num_classes, generator=None):
    graphs = list(graphs)
    if not graphs:
        raise ValueError("Cannot collate an empty image list")
    xs, labels, contexts, queries, maps, targets = [], [], [], [], [], []
    nptr, cptr, qptr = [0], [0], [0]
    for g in graphs:
        if not isinstance(g, PreparedGraph):
            g = prepare_graph(g, num_classes)
        if g.num_classes != num_classes:
            raise ValueError("Prepared supervision class count differs")
        if generator is None:
            selected = torch.arange(g.edge_label_index.shape[1])
            target = torch.zeros((len(selected), num_classes))
            target[:, 0] = 1
            target[g.positive] = g.positive_rows
        else:
            count = min(len(g.negative), 3 * len(g.positive) if len(g.positive) else 32)
            chosen = g.negative[torch.randperm(len(g.negative), generator=generator)[:count]]
            selected = torch.cat((g.positive, chosen))
            target = torch.zeros((len(selected), num_classes))
            target[: len(g.positive)] = g.positive_rows
            target[len(g.positive) :, 0] = 1
        xs.append(g.x.detach())
        labels.append(g.object_labels)
        contexts.append(g.edge_index + nptr[-1])
        queries.append(g.edge_label_index[:, selected] + nptr[-1])
        maps.append(g.context_lookup[selected] + cptr[-1])
        targets.append(target)
        nptr.append(nptr[-1] + g.num_nodes)
        cptr.append(cptr[-1] + g.edge_index.shape[1])
        qptr.append(qptr[-1] + len(selected))
    return {
        "x": torch.cat(xs),
        "object_labels": torch.cat(labels),
        "context_pairs": torch.cat(contexts, 1),
        "query_pairs": torch.cat(queries, 1),
        "query_context": torch.cat(maps),
        "targets": torch.cat(targets),
        "node_ptr": torch.tensor(nptr),
        "context_ptr": torch.tensor(cptr),
        "query_ptr": torch.tensor(qptr),
        "groups": groups_for(cptr, qptr, num_classes),
    }


def model_inputs(batch, device="cpu"):
    return {
        "batch": ModelBatch(
            *(
                batch[name].to(device)
                for name in ("x", "object_labels", "context_pairs", "query_pairs", "query_context")
            ),
            tuple(group.to(device) for group in batch["groups"]),
        )
    }
