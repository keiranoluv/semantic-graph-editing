"""Exact structured edit evaluation on fixed, already aligned entity IDs."""

from collections import Counter

from src.editing.operations import KINDS, apply_operations


def _metrics(tp, fp, fn):
    # Entirely absent types are undefined, rather than reported as perfect.
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


def evaluate_edits(examples):
    """Each row: source, target, expected operations, predicted operations.

    GT must be a valid operation set that reconstructs target. Duplicate predictions
    are false positives. Invalid predicted transactions are reported separately;
    their individual correct operations can still contribute to operation recall."""
    counts = {kind: Counter() for kind in KINDS}
    exact = valid = target_matches = total = 0
    for row in examples:
        source, target = row["source"], row["target"]
        expected, predicted = tuple(row["expected"]), tuple(row["predicted"])
        if len(set(expected)) != len(expected) or apply_operations(source, expected) != target:
            raise ValueError("Invalid GT edits or target reconstruction")
        truth, prediction = Counter(expected), Counter(predicted)
        matched, extra, missing = truth & prediction, prediction - truth, truth - prediction
        for kind in KINDS:
            counts[kind].update(
                tp=sum(n for op, n in matched.items() if op.kind == kind),
                fp=sum(n for op, n in extra.items() if op.kind == kind),
                fn=sum(n for op, n in missing.items() if op.kind == kind),
            )
        total += 1
        exact += truth == prediction
        try:
            predicted_target = apply_operations(source, predicted)
        except ValueError:
            continue
        valid += 1
        target_matches += predicted_target == target
    per_type = {kind: _metrics(c["tp"], c["fp"], c["fn"]) for kind, c in counts.items()}
    supported = [v["f1"] for v in per_type.values() if v["support"]]
    return {
        "protocol": "graph_edit_exact_payload_aligned_ids_v1",
        "examples": total,
        "per_type": per_type,
        "micro": _metrics(*(sum(c[k] for c in counts.values()) for k in ("tp", "fp", "fn"))),
        "macro_f1_supported_types": sum(supported) / len(supported) if supported else None,
        "exact_operation_set_accuracy": exact / total if total else None,
        "valid_transaction_rate": valid / total if total else None,
        "exact_target_graph_accuracy": target_matches / total if total else None,
    }
