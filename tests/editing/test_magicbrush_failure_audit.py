"""Rejected empty graph transactions cannot become no-op ground truth."""

from tools.audit_magicbrush_grounded_v2 import summarize


def test_rejected_empty_and_missing_transactions_are_unassigned_not_noops():
    report = summarize(
        [
            {
                "status": "needs_review",
                "operations": [],
                "problems": ["requested_change_not_verified"],
                "passed_automatic_filters": False,
            },
            {"status": "unsupported", "problems": ["no_grounded_source_objects"]},
        ]
    )
    assert report["empty_operation_pairs"] == 1
    assert report["pairs_without_graph_transaction"] == 1
    assert report["nonempty_operation_pairs"] == 0
    assert report["verified_noop_pairs"] == 0
    assert report["passed_automatic_filters"] == 0
