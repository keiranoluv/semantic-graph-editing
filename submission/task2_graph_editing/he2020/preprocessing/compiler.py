"""Graph-to-operations labels, strict transaction/codec/budget gates."""

from collections import Counter

from common.editing.operations import KINDS, apply_operations, diff_aligned_graphs
from task2_graph_editing.magicbrush.sequence_model.codec import canonical_source, decode_actions, encode_actions, normalize_label
from task2_graph_editing.he2020.preprocessing.alignment import align, attribute_policy, canonical_raw, validate_raw
from task2_graph_editing.he2020.preprocessing.dataset import ALIGNMENT, POLICY, VERSION

STAGES = (
    "input_availability",
    "graph_parsing",
    "schema_conversion",
    "node_alignment",
    "edge_alignment",
    "graph_diff",
    "transaction_validity",
    "codec_compatibility",
    "candidate_acceptance",
)


def compile_record(record, policy="A"):
    result = {
        **record,
        "adapter_version": VERSION,
        "conversion_policy": POLICY if policy == "A" else "B_refuse_unproven_attribute_ownership_v1",
        "alignment_algorithm": ALIGNMENT,
        "status": "rejected",
        "conversion_status": "rejected",
        "human_review_status": "pending",
        "eligible_for_training": False,
        "lossy_conversion": False,
        "problems": [],
        "stages": {k: None for k in STAGES},
        "operations": None,
    }
    stage = "input_availability"
    try:
        if record.get("load_errors"):
            raise ValueError(";".join(record["load_errors"]))
        if record.get("raw_source") is None or record.get("raw_target") is None or not record["instruction"].strip():
            raise ValueError("missing_graph_or_instruction")
        result["stages"][stage] = True
        stage = "graph_parsing"
        validate_raw(record["raw_source"])
        validate_raw(record["raw_target"])
        result["stages"][stage] = True
        stage = "schema_conversion"
        left, lp = attribute_policy(record["raw_source"], policy)
        right, rp = attribute_policy(record["raw_target"], policy)
        result["attribute_analysis"] = {"source": lp, "target": rp}
        result["stages"][stage] = True
        source, _ = canonical_raw(left)
        target, _ = canonical_raw(right)
        result["independently_converted_source"] = source.to_dict()
        result["independently_converted_target"] = target.to_dict()
        result["sizes"] = {
            "raw_source_nodes": len(left["nodes"]),
            "raw_source_edges": len(left["edges"]),
            "raw_target_nodes": len(right["nodes"]),
            "raw_target_edges": len(right["edges"]),
            "source_nodes": len(source.nodes),
            "source_edges": len(source.edges),
            "target_nodes": len(target.nodes),
            "target_edges": len(target.edges),
        }
        result["stages"][stage] = True
        stage = "node_alignment"
        source, target, evidence = align(left, right)
        result["alignment"] = evidence
        result["stages"][stage] = True
        stage = "edge_alignment"
        result["stages"][stage] = True
        result["aligned_source_graph"] = source.to_dict()
        result["aligned_target_graph"] = target.to_dict()
        stage = "graph_diff"
        ops = diff_aligned_graphs(source, target)
        result["stages"][stage] = True
        result["operation_count"] = len(ops)
        result["operation_support"] = {k: sum(o.kind == k for o in ops) for k in KINDS}
        result["no_graph_changes"] = not ops
        stage = "transaction_validity"
        if apply_operations(source, ops) != target:
            raise ValueError("transaction_does_not_reconstruct_target")
        result["stages"][stage] = True
        stage = "codec_compatibility"
        source, target, ops = normalize_label(source, target, ops)
        if source != canonical_source(canonical_raw(left)[0])[0]:
            raise ValueError("target_derived_source_identity")
        result["source_graph"], result["target_graph"] = source.to_dict(), target.to_dict()
        result["operations"] = [o.to_dict() for o in ops]
        tokens = encode_actions(ops)
        result["action_tokens_including_eos"] = len(tokens) - 1
        if len(ops) > 12:
            raise ValueError("operation_count_budget_exceeded")
        if apply_operations(source, decode_actions(tokens[1:], source)) != target:
            raise ValueError("codec_transaction_mismatch")
        result["stages"][stage] = True
        stage = "candidate_acceptance"
        problems = []
        if len(source.nodes) > 20 or len(source.edges) > 60:
            problems.append("source_graph_budget_exceeded")
        if len(target.nodes) > 20 or len(target.edges) > 60:
            problems.append("target_graph_budget_exceeded")
        if not 1 <= len(ops) <= 12:
            problems.append("no_graph_changes" if not ops else "operation_count_budget_exceeded")
        if len(tokens) - 1 > 2048:
            problems.append("action_token_budget_exceeded")
        if problems:
            result["problems"].extend(problems)
            result["stages"][stage] = False
        else:
            result["stages"][stage] = True
            result.update(status="candidate", conversion_status="candidate", eligible_for_training=True)
    except (ValueError, TypeError, KeyError) as error:
        result["stages"][stage] = False
        result["problems"].append(str(error))
    return result


def operation_counts(records):
    counts = Counter()
    evaluated = 0
    for row in records:
        if "operation_support" in row:
            counts.update(row["operation_support"])
            evaluated += 1
    return {k: counts[k] if evaluated else None for k in KINDS}
