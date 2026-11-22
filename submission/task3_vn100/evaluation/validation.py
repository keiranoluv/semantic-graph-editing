"""Validate task membership before reusing graph scores."""

from task3_vn100.evaluation.core import case_header


def validate_stage(pack, records, binding):
    if pack.get("evaluation_eligible") is not True or pack.get("stage_binding") != binding:
        raise ValueError("wrong_stage_protocol_or_smoke_not_eligible")
    if [r["task_id"] for r in pack["rows"]] != [r["task"]["task_id"] for r in records]:
        raise ValueError("stage_task_membership_changed")
    for row, record in zip(pack["rows"], records, strict=True):
        if any(row.get(k) != v for k, v in case_header(record).items()):
            raise ValueError("stage_case_binding_mismatch")
