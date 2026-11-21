"""Source-only identities and a fixed UTF-8 action vocabulary (no val vocabulary)."""

import json
from dataclasses import replace

from src.editing.evaluation import evaluate_edits
from src.editing.operations import KINDS, Edge, EditGraph, Node, Operation, apply_operations

PAD, BOS, EOS, VOCAB_SIZE = 0, 1, 2, 259


def canonical_source(source):
    """Stable source IDs require no target, boxes or editing labels."""
    nodes = {n.id: f"n{i+1:03d}" for i, n in enumerate(source.nodes)}
    edges = {e.id: f"e{i+1:03d}" for i, e in enumerate(source.edges)}
    return remap_graph(source, nodes, edges), nodes, edges


def remap_graph(graph, nodes, edges):
    return EditGraph(
        tuple(replace(n, id=nodes.get(n.id, n.id)) for n in graph.nodes),
        tuple(
            replace(
                e,
                id=edges.get(e.id, e.id),
                subject=nodes.get(e.subject, e.subject),
                object=nodes.get(e.object, e.object),
            )
            for e in graph.edges
        ),
    )


def remap_operations(operations, nodes, edges):
    def entity(value):
        if value is None:
            return None
        if isinstance(value, Node):
            return replace(value, id=nodes.get(value.id, value.id))
        return replace(
            value,
            id=edges.get(value.id, value.id),
            subject=nodes.get(value.subject, value.subject),
            object=nodes.get(value.object, value.object),
        )

    return tuple(Operation(op.kind, entity(op.before), entity(op.after)) for op in operations)


def canonical_actions(source, operations):
    """Canonicalize NEW entities independently on each side of evaluation.

    Existing IDs stay fixed. New-node keys use semantic payload and incident
    relations to existing IDs/new semantic payloads, never the reference target.
    Unresolved symmetric keys are explicitly unsupported rather than oracle matched.
    """
    operations = tuple(operations)
    target = apply_operations(source, operations)
    old_nodes, old_edges = {n.id for n in source.nodes}, {e.id for e in source.edges}
    added = [n for n in target.nodes if n.id not in old_nodes]
    by_id = {n.id: n for n in target.nodes}

    def endpoint(identifier):
        return (
            ("existing", identifier)
            if identifier in old_nodes
            else ("new", by_id[identifier].label, by_id[identifier].attributes)
        )

    def key(node):
        incident = []
        for edge in target.edges:
            if edge.subject == node.id:
                incident.append(("out", edge.predicate, endpoint(edge.object)))
            if edge.object == node.id:
                incident.append(("in", edge.predicate, endpoint(edge.subject)))
        return node.label, node.attributes, tuple(sorted(incident))

    keys = [key(n) for n in added]
    if len(set(keys)) != len(keys):
        raise ValueError("unsupported_ambiguous_new_instance_identity")
    node_ids = {n.id: f"newn{i+1:03d}" for i, n in enumerate(sorted(added, key=key))}
    interim = remap_graph(target, node_ids, {})
    new_edges = sorted(
        (e for e in interim.edges if e.id not in old_edges), key=lambda e: (e.subject, e.predicate, e.object)
    )
    edge_ids = {e.id: f"newe{i+1:03d}" for i, e in enumerate(new_edges)}
    result = remap_operations(operations, node_ids, edge_ids)
    if apply_operations(source, result) != remap_graph(target, node_ids, edge_ids):
        raise AssertionError("Identity normalization changed the transaction")
    return tuple(sorted(result, key=lambda op: (KINDS.index(op.kind), op.entity_id)))


def normalize_label(source, target, operations):
    if apply_operations(source, operations) != target:
        raise ValueError("Label does not reconstruct target")
    normalized, nodes, edges = canonical_source(source)
    # Avoid accidental collisions between arbitrary original new IDs and source IDs.
    for i, node in enumerate(n for n in target.nodes if n.id not in nodes):
        nodes[node.id] = f"rawnewn{i:03d}"
    for i, edge in enumerate(e for e in target.edges if e.id not in edges):
        edges[edge.id] = f"rawnewe{i:03d}"
    actions = canonical_actions(normalized, remap_operations(operations, nodes, edges))
    return normalized, apply_operations(normalized, actions), actions


def encode_actions(operations):
    rows = []
    for op in operations:
        row = [op.kind, op.entity_id]
        if op.after is not None:
            if isinstance(op.after, Node):
                row += [op.after.label, list(op.after.attributes)]
            else:
                row += [op.after.subject, op.after.predicate, op.after.object]
        rows.append(row)
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return [BOS, *(b + 3 for b in payload), EOS]


def decode_actions(tokens, source):
    """Before payloads come from the SOURCE, never reference labels or targets."""
    tokens = list(tokens)
    if not tokens or tokens[-1] != EOS or any(type(t) is not int or not 3 <= t < VOCAB_SIZE for t in tokens[:-1]):
        raise ValueError("Missing EOS or invalid action byte")
    rows = json.loads(bytes(t - 3 for t in tokens[:-1]).decode("utf-8"))
    if not isinstance(rows, list) or len(rows) > 12:
        raise ValueError("Action list must contain at most 12 operations")
    nodes, edges = {n.id: n for n in source.nodes}, {e.id: e for e in source.edges}
    result = []
    for row in rows:
        if not isinstance(row, list) or len(row) < 2 or row[0] not in KINDS or not isinstance(row[1], str):
            raise ValueError("Invalid action schema")
        kind, identifier = row[:2]
        action, entity = kind.split("_")
        table = nodes if entity == "node" else edges
        expected_size = 2 if action == "delete" else 4 if entity == "node" else 5
        if len(row) != expected_size:
            raise ValueError("Invalid action arity")
        before = table.get(identifier) if action != "add" else None
        if action != "add" and before is None:
            raise ValueError("Unknown source entity")
        after = None
        if action != "delete":
            if entity == "node":
                if (
                    not isinstance(row[2], str)
                    or not isinstance(row[3], list)
                    or any(not isinstance(x, str) for x in row[3])
                ):
                    raise ValueError("Invalid node payload")
                after = Node(identifier, row[2], tuple(row[3]))
            else:
                if any(not isinstance(v, str) for v in row[2:]):
                    raise ValueError("Invalid edge payload")
                after = Edge(identifier, *row[2:])
        result.append(Operation(kind, before, after))
    return tuple(result)


def score_predictions(examples, token_predictions):
    if len(examples) != len(token_predictions):
        raise ValueError("Prediction count mismatch")
    rows, errors = [], []
    for example, tokens in zip(examples, token_predictions, strict=True):
        error = None
        try:
            predicted = decode_actions(tokens, example.source)
        except (ValueError, TypeError, KeyError, UnicodeError) as exc:
            predicted, error = (), f"parse: {exc}"
        if error is None:
            try:
                predicted = canonical_actions(example.source, predicted)
            except ValueError as exc:
                # Preserve parsable actions for per-operation FP/FN; transaction
                # or unresolved identity errors cannot count as exact matches.
                category = "identity" if str(exc) == "unsupported_ambiguous_new_instance_identity" else "transaction"
                error = f"{category}: {exc}"
        metrics = evaluate_edits(
            [
                {
                    "source": example.source,
                    "target": example.target,
                    "expected": example.operations,
                    "predicted": predicted,
                }
            ]
        )
        if error:
            for key in ("exact_operation_set_accuracy", "exact_target_graph_accuracy"):
                metrics[key] = 0.0
            if error.startswith("parse:"):
                metrics["valid_transaction_rate"] = 0.0
        rows.append((metrics, example.simple))
        errors.append(error)
    aggregate = {
        "examples": len(rows),
        "per_type": {},
        "failed_predictions": sum(e is not None for e in errors),
        "decode_failures": sum(bool(e and e.startswith("parse:")) for e in errors),
        "invalid_transactions": sum(bool(e and e.startswith("transaction:")) for e in errors),
        "unsupported_predicted_identities": sum(bool(e and e.startswith("identity:")) for e in errors),
    }
    for kind in KINDS:
        tp, fp, fn = (sum(row[0]["per_type"][kind][k] for row in rows) for k in ("tp", "fp", "fn"))
        aggregate["per_type"][kind] = counts_metrics(tp, fp, fn)
    aggregate["micro"] = counts_metrics(
        *(sum(r[k] for r in aggregate["per_type"].values()) for k in ("tp", "fp", "fn"))
    )
    supported = [r["f1"] for r in aggregate["per_type"].values() if r["support"]]
    aggregate["macro_f1_supported_types"] = sum(supported) / len(supported) if supported else None
    for key in ("exact_operation_set_accuracy", "valid_transaction_rate", "exact_target_graph_accuracy"):
        aggregate[key] = sum(row[0][key] for row in rows) / len(rows) if rows else None
    simple = [r for r, is_simple in rows if is_simple]
    aggregate["simple_examples"] = len(simple)
    aggregate["simple_exact_operation_set_accuracy"] = (
        sum(r["exact_operation_set_accuracy"] for r in simple) / len(simple) if simple else None
    )
    aggregate["protocol"] = "source_ids_and_independent_new_identity_canonicalization_utf8_actions_v1"
    return aggregate, errors


def counts_metrics(tp, fp, fn):
    present = tp + fp + fn > 0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "support": tp + fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0 if present else None,
        "recall": tp / (tp + fn) if tp + fn else 0.0 if present else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if present else None,
    }
