import copy

import pytest

from tools.level3_label_recheck.core import normalize


def fixture():
    values = [(True, True), (None, None), (False, None), (None, False), (False, False), (True, False)]
    records = [{"task": {"task_id": f"t{i}"}, "binding": f"b{i}"} for i in range(len(values))]
    pack = {
        "schema": "qwen-unified-review-v1",
        "cohort_sha256": "cohort",
        "reviews": [
            {
                "task_id": f"t{i}",
                "binding": f"b{i}",
                "edit_fulfillment": a,
                "preservation": b,
                "status": "pending" if a is None or b is None else "reviewed",
            }
            for i, (a, b) in enumerate(values)
        ],
    }
    return pack, records


def test_user_default_keeps_explicit_false_and_tracks_each_default():
    pack, records = fixture()
    before = copy.deepcopy(pack)
    result, failed, summary = normalize(pack, records, "cohort", blank_is_pass=True)
    assert pack == before
    assert failed == ["t2", "t3", "t4", "t5"]
    assert summary["failure_types"] == {"fulfillment_only": 1, "preservation_only": 2, "both": 1}
    assert summary["fields_defaulted"] == 4 and summary["rows_with_defaults"] == 3
    assert result["reviews"][2]["edit_fulfillment"] is False
    assert result["reviews"][2]["preservation"] is True
    assert result["normalization"]["defaulted_fields"]["t2"] == ["preservation"]
    assert all(r["status"] == "reviewed" for r in result["reviews"])


def test_no_global_automatic_pass_without_user_authorization():
    pack, records = fixture()
    with pytest.raises(ValueError, match="explicit_user"):
        normalize(pack, records, "cohort")


@pytest.mark.parametrize(
    "problem", ["missing_row", "duplicate", "wrong_binding", "missing_field", "numeric", "other_cohort"]
)
def test_incomplete_or_foreign_submission_not_silently_passed(problem):
    pack, records = fixture()
    if problem == "missing_row":
        pack["reviews"].pop()
    if problem == "duplicate":
        pack["reviews"][-1] = pack["reviews"][0]
    if problem == "wrong_binding":
        pack["reviews"][1]["binding"] = "other-image"
    if problem == "missing_field":
        del pack["reviews"][1]["preservation"]
    if problem == "numeric":
        pack["reviews"][1]["edit_fulfillment"] = 1
    if problem == "other_cohort":
        pack["cohort_sha256"] = "other"
    with pytest.raises(ValueError):
        normalize(pack, records, "cohort", blank_is_pass=True)
