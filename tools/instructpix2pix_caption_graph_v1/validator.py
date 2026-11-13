"""Repository transaction validation and downstream graph/action budgets."""

from src.editing.operations import apply_operations
from tools.edit_model_v1.codec import decode_actions, encode_actions, normalize_label


def validate(source, target, operations):
    if not 1 <= len(operations) <= 12:
        raise ValueError("operation_count_outside_1_12")
    if any(len(g.nodes) > 20 or len(g.edges) > 60 for g in (source, target)):
        raise ValueError("downstream_graph_budget")
    if apply_operations(source, operations) != target:
        raise ValueError("transaction_target_mismatch")
    source, target, operations = normalize_label(source, target, operations)
    tokens = encode_actions(operations)
    if len(tokens) - 1 > 2048:
        raise ValueError("downstream_action_budget")
    if apply_operations(source, decode_actions(tokens[1:], source)) != target:
        raise ValueError("downstream_codec_roundtrip")
    return source, target, operations
