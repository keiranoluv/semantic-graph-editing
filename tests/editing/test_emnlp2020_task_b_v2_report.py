from tools.report_emnlp2020_task_b_v2 import aggregate, classify, sanitized_cases


def test_seed_aggregation_uses_sample_std_and_preserves_unsupported_types():
    from src.editing.evaluation import evaluate_edits
    from src.editing.operations import EditGraph

    metric = evaluate_edits([{"source": EditGraph(), "target": EditGraph(), "expected": (), "predicted": ()}])
    metric["simple_exact_operation_set_accuracy"] = None
    result = aggregate([metric, metric, metric])
    assert result["exact_target_graph_accuracy"] == {"mean": 1.0, "sample_std": 0.0, "values": [1.0, 1.0, 1.0]}
    assert result["per_type"]["replace_edge"]["f1"]["mean"] is None


def test_case_sanitization_keeps_diagnosis_without_releasing_text_or_payload():
    expected = {
        "kind": "replace_node",
        "before": {"id": "n001", "label": "private"},
        "after": {"id": "n001", "label": "secret"},
    }
    case = {
        "id": "x",
        "correct": False,
        "error": None,
        "reference_OOV": True,
        "instruction": "private text",
        "expected": [expected],
        "predicted": [expected],
    }
    assert classify(case) == "wrong_payload"
    output = sanitized_cases([case])[0]
    assert output["expected_kinds"] == {"replace_node": 1}
    assert "instruction" not in output
    assert "private" not in str(output)
    assert "secret" not in str(output)
