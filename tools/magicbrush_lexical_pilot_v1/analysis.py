"""Stored-prediction diagnostics; no metric relaxation and no model execution."""
from collections import Counter

from tools.edit_model_v1.codec import canonical_actions, decode_actions


def diagnose(examples, rows):
    if [e.id for e in examples] != [r["id"] for r in rows]:
        raise ValueError("Diagnostic ID/order mismatch")
    counts = Counter()
    wrong = Counter()
    illustrations = []
    for e, row in zip(examples, rows, strict=True):
        try:
            pred = canonical_actions(e.source, decode_actions(row["tokens"], e.source))
        except (ValueError, TypeError, KeyError):
            pred = ()
        expected = Counter(o.after.label for o in e.operations if o.kind == "add_node")
        added = Counter(o.after.label for o in pred if o.kind == "add_node")
        counts["expected_added_names"] += sum(expected.values())
        counts["predicted_added_names"] += sum(added.values())
        counts["correct_added_names_ignoring_attributes_and_ids"] += sum((expected & added).values())
        wrong.update(added - expected)
        index = {(o.kind, o.entity_id): o for o in pred if not o.kind.startswith("add_")}
        for op in e.operations:
            if op.kind.startswith("add_"):
                continue
            p = index.get((op.kind, op.entity_id))
            counts["expected_existing_ops"] += 1
            counts["correct_existing_kind_and_id"] += p is not None
            counts["correct_existing_complete_payload"] += p == op
        if expected and any(w in e.instruction.lower() for w in ["laptop", "gorilla", "child", "lion"]):
            illustrations.append(
                {
                    "id": e.id,
                    "instruction": e.instruction,
                    "expected_added_names": dict(expected),
                    "predicted_added_names": dict(added),
                }
            )
    return {
        "counts": dict(counts),
        "frequent_wrong_added_names": wrong.most_common(5),
        "illustrations": illustrations,
        "scope": "name matching ignores attributes/IDs; source ID/type match ignores payload. Diagnostics only, not replacement metrics.",
    }
