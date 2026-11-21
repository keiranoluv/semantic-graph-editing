"""Full, label-free context topology and separately sampled supervision."""

import torch

from common.training.gqa_pilot import sample_pairs, soft_targets

MODEL_FIELDS = ("x", "object_labels", "node_ptr", "context_pairs", "context_ptr", "query_pairs", "query_ptr")


def validate_pairs(pairs, nodes, *, full=False):
    if pairs.dtype != torch.long or pairs.ndim != 2 or pairs.shape[0] != 2:
        raise ValueError("Expected long [2,E] pairs")
    if pairs.numel() and ((pairs < 0).any() or (pairs >= nodes).any() or (pairs[0] == pairs[1]).any()):
        raise ValueError("Invalid local pair endpoints/self-loop")
    if full and (
        pairs.shape[1] != nodes * (nodes - 1) or len(torch.unique(pairs[0] * nodes + pairs[1])) != pairs.shape[1]
    ):
        raise ValueError("Context must contain every ordered non-self pair exactly once")


def context_pairs(graph):
    """Read only full graph topology/node count; never annotation or prepared_* fields.

    Completeness validation rejects callers that accidentally pass sampled
    positive/negative topology. Ordering from the historical builder is retained.
    """
    pairs = graph.edge_index
    validate_pairs(pairs, graph.num_nodes, full=True)
    return pairs.detach().clone()


def _pack(graphs, contexts, queries):
    xs, labels, cp, qp = [], [], [], []
    node_ptr, context_ptr, query_ptr = [0], [0], [0]
    for graph, context, query in zip(graphs, contexts, queries, strict=True):
        xs.append(graph.x.detach())
        labels.append(graph.object_labels)
        cp.append(context + node_ptr[-1])
        qp.append(query + node_ptr[-1])
        node_ptr.append(node_ptr[-1] + graph.num_nodes)
        context_ptr.append(context_ptr[-1] + context.shape[1])
        query_ptr.append(query_ptr[-1] + query.shape[1])
    return {
        "x": torch.cat(xs),
        "object_labels": torch.cat(labels),
        "node_ptr": torch.tensor(node_ptr, dtype=torch.long),
        "context_pairs": torch.cat(cp, dim=1),
        "context_ptr": torch.tensor(context_ptr, dtype=torch.long),
        "query_pairs": torch.cat(qp, dim=1),
        "query_ptr": torch.tensor(query_ptr, dtype=torch.long),
    }


def inference_batch(graphs):
    """Full-candidate prediction without reading any ground-truth predicates."""
    graphs = list(graphs)
    if not graphs:
        raise ValueError("Cannot collate an empty batch")
    contexts = [context_pairs(g) for g in graphs]
    queries = []
    for graph in graphs:
        validate_pairs(graph.edge_label_index, graph.num_nodes, full=True)
        queries.append(graph.edge_label_index)
    return _pack(graphs, contexts, queries)


def collate(graphs, num_classes, generator=None, *, negative_ratio=3, empty_negatives=32):
    """Historical sampler/soft targets unchanged; generator=None means all queries.

    Context is constructed before labels are inspected. Sampling settings affect
    queries only. Pointers preserve image boundaries, offsets and query order.
    """
    graphs = list(graphs)
    if not graphs or type(num_classes) is not int or num_classes < 2:
        raise ValueError("Expected a nonempty graph list and at least two classes")
    contexts = [context_pairs(g) for g in graphs]
    queries, targets = [], []
    for graph in graphs:
        validate_pairs(graph.edge_label_index, graph.num_nodes, full=True)
        target = soft_targets(graph, num_classes)
        selected = (
            sample_pairs(target, generator, negative_ratio, empty_negatives)
            if generator is not None
            else torch.arange(len(target), dtype=torch.long)
        )
        queries.append(graph.edge_label_index[:, selected])
        targets.append(target[selected])
    return {**_pack(graphs, contexts, queries), "targets": torch.cat(targets)}


def model_inputs(batch, device="cpu"):
    """Explicit allowlist: targets and all predicate annotations stay outside forward."""
    return {name: batch[name].to(device) for name in MODEL_FIELDS}
