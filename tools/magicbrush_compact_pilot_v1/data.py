"""Keep all validation IDs while separating known schema gaps from scored labels."""
from dataclasses import replace

from tools.edit_model_v1.codec import normalize_label
from tools.magicbrush_compact_v1.build import example_dict
from tools.magicbrush_compact_v1.projection import project_example
from tools.magicbrush_experiment_v1.evaluation import score_rows
from tools.prepare_magicbrush import digest


def load_validation(payload, originals, policy):
    rows = [r for r in payload["records"] if r["split"] == "val"]
    if len({r["id"] for r in rows}) != len(rows) or {r["id"] for r in rows} != {e.id for e in originals}:
        raise ValueError("Validation membership changed")
    index = {r["id"]: r for r in rows}
    validation = []
    for old in originals:
        new, audit = project_example(old, policy)
        expected = {**example_dict(new), "original_label_sha256": digest(example_dict(old)), "flags": audit["flags"]}
        if digest(index[old.id]) != digest(expected) or any(f != "no_core_edit" for f in audit["flags"]):
            raise ValueError("Validation projection differs or has unsupported identity")
        source, target, operations = normalize_label(new.source, new.target, new.operations)
        validation.append(replace(new, source=source, target=target, operations=operations))
    return validation


def partition(validation):
    scored = [e for e in validation if e.operations]
    gaps = [e for e in validation if not e.operations]
    if not scored:
        raise ValueError("No actionable validation labels")
    return scored, gaps


def evaluation_report(validation, rows):
    if [e.id for e in validation] != [r["id"] for r in rows]:
        raise ValueError("Full validation prediction IDs/order changed")
    scored, gaps = partition(validation)
    by_id = {r["id"]: r for r in rows}
    main = [by_id[e.id] for e in scored]
    metrics = score_rows(scored, main)
    raw = score_rows(scored, [{"id": r["id"], "tokens": r["raw_tokens"], "error": r["raw_error"]} for r in main])
    return {
        "scope": "Primary metrics on fixed actionable compact validation labels; schema gaps retained and not scored as successful empty edits",
        "validation_pairs": len(validation),
        "scored_pairs": len(scored),
        "schema_gap_pairs": len(gaps),
        "schema_coverage": len(scored) / len(validation),
        "schema_gap_ids": [e.id for e in gaps],
        "constrained": metrics,
        "raw": raw,
        "conservative_exact_graph_over_all_pairs": metrics["exact_target_graph_accuracy"]
        * len(scored)
        / len(validation),
        "conservative_exact_definition": "Correct compact graphs on actionable cases divided by ALL validation pairs; schema-gap cases receive no success credit. Not original full-detail graph accuracy.",
        "gap_diagnostics": [
            {
                "id": e.id,
                "instruction": e.instruction,
                "predicted_operation_count": by_id[e.id]["operation_count"],
                "error": by_id[e.id]["error"],
                "correctness_assessed": False,
            }
            for e in gaps
        ],
    }
