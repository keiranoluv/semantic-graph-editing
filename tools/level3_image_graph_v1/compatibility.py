"""Task B audit only: never silently flatten attributes or symmetrize relations."""

import json

from src.editing.operations import EditGraph
from tools.edit_model_v1.codec import canonical_source

from .core import sha256


def load_task_b(path):
    import torch

    saved = torch.load(path, map_location="cpu", weights_only=True)
    if not {"vocabulary", "state_dict", "config", "experiment_identity"} <= saved.keys():
        raise ValueError("expected_emnlp_task_b_v4_checkpoint")
    tokens = saved["vocabulary"]
    if not isinstance(tokens, list) or not all(isinstance(t, str) for t in tokens) or len(set(tokens)) != len(tokens):
        raise ValueError("invalid_task_b_vocabulary")
    config, state = saved["config"], saved["state_dict"]
    if (
        state["node_identity.weight"].shape[0] != config["max_nodes"]
        or state["edge_identity.weight"].shape[0] != config["max_edges"]
        or state["output.weight"].shape[0] != len(tokens) + 4
    ):
        raise ValueError("task_b_config_does_not_match_weights")
    return {
        "checkpoint_sha256": sha256(path),
        "vocabulary": tokens,
        "config": saved["config"],
        "experiment_identity": saved["experiment_identity"],
    }


def audit_graph(graph, checkpoint=None):
    max_nodes = checkpoint["config"]["max_nodes"] if checkpoint else 7
    max_edges = checkpoint["config"]["max_edges"] if checkpoint else 42
    normalized, node_ids, edge_ids = canonical_source(graph)
    if EditGraph.from_dict(normalized.to_dict()) != normalized:
        raise ValueError("codec_roundtrip_failed")
    triples = {(e.subject, e.predicate, e.object) for e in graph.edges}
    one_way = [e.id for e in graph.edges if (e.object, e.predicate, e.subject) not in triples]
    fields = {json.dumps(n.label, ensure_ascii=False, separators=(",", ":")) for n in graph.nodes}
    fields |= {json.dumps(list(n.attributes), ensure_ascii=False, separators=(",", ":")) for n in graph.nodes}
    fields |= {json.dumps(e.predicate, ensure_ascii=False, separators=(",", ":")) for e in graph.edges}
    oov = sorted(fields - set(checkpoint["vocabulary"])) if checkpoint else None
    reasons = [
        "visual_domain_transfer_not_validated",
        "official_attribute_nodes_and_att_links_differ_from_node_attributes",
    ]
    if one_way:
        reasons.append("directed_visual_relations_differ_from_reciprocal_undirected_training_links")
    if not graph.nodes:
        reasons.append("empty_graph_not_an_observed_scene")
    if len(graph.nodes) > max_nodes or len(graph.edges) > max_edges:
        reasons.append("task_b_source_budget_exceeded")
    if any(n.attributes for n in graph.nodes):
        reasons.append("inline_attributes_not_seen_in_policy_A_training")
    if oov:
        reasons.append("source_payload_not_fully_expressible_by_action_vocabulary")
    if checkpoint is None:
        reasons.append("checkpoint_vocabulary_not_checked")
    return {
        "editgraph_schema_valid": True,
        "codec_roundtrip_valid": True,
        "within_task_b_input_budget": 0 < len(graph.nodes) <= max_nodes and len(graph.edges) <= max_edges,
        "task_b_limits": {
            "nodes": max_nodes,
            "edges": max_edges,
            "basis": "checkpoint_config" if checkpoint else "v4_config_defaults_unverified",
        },
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
        "one_way_edge_ids": one_way,
        "inline_attribute_nodes": [n.id for n in graph.nodes if n.attributes],
        "source_payload_oov": oov,
        "source_payload_oov_scope": "output lexical fields, not CLIP input vocabulary; target edits unknown",
        "canonical_id_map": {"nodes": node_ids, "edges": edge_ids},
        "direct_model_use_approved": False,
        "semantic_compatibility": "not_established",
        "reasons": reasons,
        "graph_transformed": False,
    }


def audit_records(records, checkpoint=None):
    return {
        "protocol": "level3-task-b-compatibility-v1",
        "checkpoint_sha256": checkpoint["checkpoint_sha256"] if checkpoint else None,
        "checkpoint_vocabulary_size": len(checkpoint["vocabulary"]) if checkpoint else None,
        "records": {
            r["candidate_id"]: audit_graph(EditGraph.from_dict(r["source_graph"]), checkpoint) for r in records
        },
        "direct_model_use_approved": False,
        "evidence": [
            "tools/emnlp2020_graph_modification_v1/alignment.py:attribute_policy,canonical_raw",
            "tools/emnlp2020_graph_modification_v1/compiler.py:compile_record",
            "tools/emnlp2020_task_b_v2/model.py:ActionVocabulary",
            "tools/edit_model_v1/data.py:feature_texts",
        ],
        "required_next_step": "Version an explicit representation adapter and validate paired human-reviewed edits before transfer claims. Do not infer attribute ownership from undirected att links.",
    }
