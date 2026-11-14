"""CPU audit of sealed easy/copy predictions; no model inference or test labels."""
from collections import Counter
from dataclasses import replace
from pathlib import Path

from src.editing.operations import EditGraph, apply_operations, diff_aligned_graphs
from tools.edit_model_v1.codec import canonical_actions, decode_actions, encode_actions, score_predictions
from tools.emnlp2020_graph_modification_v1.dataset import read, sha256, write
from tools.magicbrush_easy_copy_v1.candidates import candidates
from tools.magicbrush_easy_copy_v1.projection import omit_added_appearance
from tools.magicbrush_easy_copy_v1.runner import completed, preflight
from tools.magicbrush_experiment_v3.vocabulary import build_vocabulary
from tools.magicbrush_lexical_pilot_v1.analysis import diagnose
from tools.prepare_magicbrush import ROOT, digest


def project_prediction(source, instruction, tokens):
    """Apply the already frozen annotation rule to a PREDICTED graph only.

    This offline diagnostic cannot consult a reference target, expected actions,
    or reference multiplicity. It does not replace the sealed main predictions.
    """
    operations = decode_actions(tokens, source)
    target = apply_operations(source, operations)
    old = {n.id for n in source.nodes}
    names = Counter(n.label for n in target.nodes if n.id not in old)
    enabled = omit_added_appearance(source, instruction)
    nodes = tuple(
        replace(n, attributes=()) if enabled and n.id not in old and names[n.label] == 1 else n for n in target.nodes
    )
    projected = EditGraph(nodes, target.edges)
    return encode_actions(diff_aligned_graphs(source, projected))[1:]


def payload_diagnostics(examples, rows, vocab):
    counts = Counter()
    cases = []
    for e, row in zip(examples, rows, strict=True):
        pred = canonical_actions(e.source, decode_actions(row["tokens"], e.source))
        expected = Counter(o.after.label for o in e.operations if o.kind == "add_node")
        actual = Counter(o.after.label for o in pred if o.kind == "add_node")
        gt_payload = Counter((o.after.label, o.after.attributes) for o in e.operations if o.kind == "add_node")
        p_payload = Counter((o.after.label, o.after.attributes) for o in pred if o.kind == "add_node")
        counts["added_name_and_attributes_match_ignoring_id"] += sum((gt_payload & p_payload).values())
        empty_gt = Counter(o.after.label for o in e.operations if o.kind == "add_node" and not o.after.attributes)
        nonempty_pred = Counter(o.after.label for o in pred if o.kind == "add_node" and o.after.attributes)
        counts["expected_attribute_free_added_nodes"] += sum(empty_gt.values())
        counts["correct_name_but_extraneous_attributes_on_attribute_free_reference"] += sum(
            (empty_gt & nonempty_pred).values()
        )
        representable_replace = 0
        for o in e.operations:
            if o.kind != "replace_node":
                continue
            match = next((p for p in pred if p.kind == o.kind and p.entity_id == o.entity_id), None)
            counts["reference_replace_node"] += 1
            counts["replace_node_correct_kind_id"] += match is not None
            counts["replace_node_correct_kind_id_label"] += match is not None and match.after.label == o.after.label
            attrs = set(o.after.attributes) - set(o.before.attributes)
            name = (
                o.before.label == o.after.label
                or o.after.label in vocab["labels"]
                or o.after.label in candidates(e.instruction)
            )
            representable_replace += name and attrs <= set(vocab["attributes"])
        counts["replace_node_payload_representable_with_copy"] += representable_replace
        if e.id in {"magicbrush:338078:1", "magicbrush:338078:3", "magicbrush:357633:1", "magicbrush:484847:1"}:
            cases.append(
                {
                    "id": e.id,
                    "instruction": e.instruction,
                    "expected": [o.to_dict() for o in e.operations],
                    "predicted": [o.to_dict() for o in pred],
                }
            )
        if any(name not in vocab["labels"] for name in expected):
            for name, count in expected.items():
                if name not in vocab["labels"]:
                    counts["oov_added_names_correct"] += min(count, actual[name])
    return {"counts": dict(counts), "illustrations": cases}


def main():
    cfg, examples, identity, data_report = preflight()
    output = ROOT / cfg["output"]
    summary = read(output / "summary.json")
    cache = ROOT / cfg["text_cache"]
    meta = read(cache / "manifest.json")
    assert read(output / "experiment.json") == meta["identity"] == identity
    assert sha256(cache / "vectors.pt") == meta["vectors_sha256"]
    training_identity = {**identity, "frozen_vectors_sha256": meta["vectors_sha256"]}
    assert read(output / "training-identity.json") == training_identity
    assert summary["identity_sha256"] == digest(training_identity)
    assert summary["test_used"] is False
    assert read(output / "progress.json")["stage"] == "complete"
    audit = {
        "scope": "CPU rescoring of sealed validation predictions, no new model inference, no test targets",
        "training_code_commit": "0597661",
        "seed": cfg["seed"],
        "training_identity_sha256": digest(training_identity),
        "validation_pairs": len(examples["val"]),
        "train_pairs": len(examples["train"]),
        "sources_sha256": {
            str(p.relative_to(ROOT)): sha256(p)
            for p in [output / "summary.json", cache / "manifest.json", Path(__file__).resolve()]
        },
        "checks": [
            "preflight verifies locked source and label provenance",
            "frozen feature identity/hash verified",
            "four completion receipts verified",
            "raw and constrained metrics independently rescored",
            "all prediction IDs/order and error annotations checked",
            "stored name/ID diagnostics recomputed",
        ],
        "models": {},
        "copy_candidate_coverage": data_report["copy_name_coverage"]["val"],
        "offline_schema_projection_scope": "Already frozen annotation rule applied to predicted graphs using only source/instruction/predictions, including predicted duplicate-name guard; same 67-case references and metric. Diagnostic only, not a new trained result or target-aware correction.",
        "gpu_jobs_started_by_audit": False,
        "test_used": False,
    }
    # Counterfactual availability only: do not refit or change any model.
    vocab = identity["vocabulary"]
    all_context = build_vocabulary(examples["train"])
    availability = Counter()
    unavailable = []
    for e in examples["val"]:
        for op in e.operations:
            if op.kind != "replace_node":
                continue
            name_ok = (
                op.before.label == op.after.label
                or op.after.label in vocab["labels"]
                or op.after.label in candidates(e.instruction)
            )
            attrs = set(op.after.attributes) - set(op.before.attributes)
            missing = attrs - set(vocab["attributes"])
            availability["reference_replace_node"] += 1
            availability["name_unavailable_even_with_copy"] += not name_ok
            availability["added_attribute_unavailable"] += bool(missing)
            availability["representable_current"] += name_ok and not missing
            availability[
                "representable_if_all_train_context_attribute_prototypes_restored"
            ] += name_ok and attrs <= set(all_context["attributes"])
            if missing:
                unavailable.append(
                    {
                        "id": e.id,
                        "entity_id": op.entity_id,
                        "missing_attributes": sorted(missing),
                        "available_in_train_context": sorted(missing & set(all_context["attributes"])),
                    }
                )
    audit["replace_payload_availability"] = {
        "counts": dict(availability),
        "cases": unavailable,
        "scope": "Representability diagnostic, not achieved accuracy; all-context vocabulary uses train only",
    }
    decoded = {}
    for arm in cfg["queue"]:
        variant, kind = arm["variant"], arm["model"]
        key = variant + "/" + kind
        directory = output / variant / kind / str(cfg["seed"])
        run_id = {
            "experiment_sha256": digest({**training_identity, "variant": variant}),
            "model": kind,
            "seed": cfg["seed"],
        }
        result, diag = completed(directory, run_id)
        assert result["identity"] == run_id
        rows = read(directory / "predictions-val.json")
        assert [r["id"] for r in rows] == [e.id for e in examples["val"]]
        for tokens, field, errors_field in [
            ("tokens", "validation", "error"),
            ("raw_tokens", "raw_validation", "raw_error"),
        ]:
            measured, errors = score_predictions(examples["val"], [r[tokens] for r in rows])
            assert errors == [r[errors_field] for r in rows]
            assert measured == result[field] == summary["models"][key][field]
        assert digest(diagnose(examples["val"], rows)) == digest(diag) == digest(summary["models"][key]["diagnostics"])
        assert result["selected_epoch"] == summary["models"][key]["epoch"]
        assert result["epochs_completed"] == summary["models"][key]["epochs_completed"]
        projected = [
            project_prediction(e.source, e.instruction, r["tokens"]) for e, r in zip(examples["val"], rows, strict=True)
        ]
        projected_metrics, projected_errors = score_predictions(examples["val"], projected)
        write(
            ROOT / cfg["reports"] / (variant + "-" + kind + "-schema-projection.json"),
            [
                {"id": e.id, "tokens": t, "error": err}
                for e, t, err in zip(examples["val"], projected, projected_errors, strict=True)
            ],
        )
        decoded[key] = [
            canonical_actions(e.source, decode_actions(r["tokens"], e.source))
            for e, r in zip(examples["val"], rows, strict=True)
        ]
        audit["models"][key] = {
            "selected_epoch": result["selected_epoch"],
            "epochs_completed": result["epochs_completed"],
            "stopping_reason": result["stopping_reason"],
            "validation": result["validation"],
            "raw_validation": result["raw_validation"],
            "diagnostics": diag,
            "payload_diagnostics": payload_diagnostics(examples["val"], rows, identity["vocabulary"]),
            "offline_schema_projection": projected_metrics,
            "completion_receipt_sha256": sha256(directory / "completed-receipt.json"),
            "loss_history": [
                {k: r[k] for k in ("epoch", "train_structured_loss", "validation_structured_loss") if k in r}
                for r in result["history"]
                if r["epoch"] in (1, result["selected_epoch"], result["epochs_completed"])
            ],
        }
    audit["comparison"] = {}
    for kind in ("mlp", "gnn"):
        changes = []
        for e, before, after in zip(examples["val"], decoded["no_copy/" + kind], decoded["copy/" + kind], strict=True):
            expected = Counter(o.after.label for o in e.operations if o.kind == "add_node")
            a, b = (Counter(o.after.label for o in ops if o.kind == "add_node") for ops in (before, after))
            gain = sum((expected & b).values()) - sum((expected & a).values())
            if gain:
                changes.append(
                    {
                        "id": e.id,
                        "instruction": e.instruction,
                        "expected": dict(expected),
                        "no_copy": dict(a),
                        "copy": dict(b),
                        "net_name_gain": gain,
                    }
                )
        a, b = (audit["models"][v + "/" + kind]["validation"]["micro"]["f1"] for v in ("no_copy", "copy"))
        assert summary["comparison"][kind] == {"no_copy_f1": a, "copy_f1": b}
        audit["comparison"][kind] = {"delta_f1_percentage_points": 100 * (b - a), "name_gain_loss_cases": changes}
    write(ROOT / cfg["reports"] / "result-audit.json", audit)
    for key, r in audit["models"].items():
        print(
            key,
            "F1",
            r["validation"]["micro"]["f1"],
            "raw validity",
            r["raw_validation"]["valid_transaction_rate"],
            "payload",
            r["payload_diagnostics"]["counts"],
            "offline projection F1/exact",
            r["offline_schema_projection"]["micro"]["f1"],
            r["offline_schema_projection"]["exact_target_graph_accuracy"],
        )


if __name__ == "__main__":
    main()
