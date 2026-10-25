"""Apply Toan's direct focus decisions and approval of the other AI proposals.

The approval declaration records the user's exact instruction and pins the
source proposal, focus list, and selected evidence. This produces a new review
version; it never changes an earlier draft or KG.
"""

import argparse
import csv
import hashlib
import io
import json
import re
import tempfile
from pathlib import Path

from scripts.build_conceptnet_graph import write_outputs
from scripts.prepare_conceptnet_review import REVIEW_FIELDS, _evidence, _source, check
from scripts.suggest_conceptnet_review import FIELDS

SELECTION_FIELDS = ("class_id", "selected_uri", "evidence_reference", "rationale")
DIRECT_FIELDS = ("class_id", "label", "decision", "selected_uri", "reviewer", "evidence_reference", "rationale")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _csv(data: bytes, fields: tuple[str, ...]) -> list[dict]:
    reader = csv.DictReader(io.StringIO(data.decode("utf-8")))
    if reader.fieldnames != list(fields):
        raise ValueError(f"Unexpected CSV columns: {reader.fieldnames}")
    return list(reader)


def _csv_bytes(rows: list[dict], fields: tuple[str, ...]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def apply_review(
    kg_dir: Path, source_dir: Path, approval_file: Path, selection_file: Path
) -> tuple[dict[str, bytes], dict]:
    if check(kg_dir, source_dir) != {"accept": 0, "reject": 0, "defer": 0, "pending": 150}:
        raise ValueError("Expected the untouched ConceptNet review worksheet")
    proposal_bytes = (source_dir / "review_suggestions.ai-draft.csv").read_bytes()
    focus_bytes = (source_dir / "review_focus.ai-draft.csv").read_bytes()
    proposal_info = json.loads((source_dir / "review_suggestions.ai-draft.json").read_text(encoding="utf-8"))
    approval_bytes = approval_file.read_bytes()
    approval = json.loads(approval_bytes)
    selection_bytes = selection_file.read_bytes()
    accepted_ids = approval.get("accepted_focus_class_ids")
    approver = approval.get("approver")
    if (
        approval.get("schema_version") != 1
        or approval.get("approval_scope") != "conceptnet_ai_134_plus_16_direct_focus"
        or approver != proposal_info.get("requested_human_approver")
        or not isinstance(approver, str)
        or approver.strip().casefold() in {"", "ai", "openai", "chatgpt", "assistant", "llm"}
        or approval.get("proposal_sha256") != sha256(proposal_bytes)
        or approval.get("focus_sha256") != sha256(focus_bytes)
        or approval.get("selection_sha256") != sha256(selection_bytes)
        or approval.get("approve_non_focus_ai_proposals") is not True
        or approval.get("reject_remaining_focus") is not True
        or not isinstance(accepted_ids, list)
        or any(type(i) is not int for i in accepted_ids)
        or accepted_ids != sorted(set(accepted_ids))
        or not isinstance(approval.get("statement"), str)
        or len(approval["statement"].strip()) < 20
        or not re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", approval.get("approved_at_utc", ""))
        or proposal_info.get("status") != "AI_PROVISIONAL_NOT_HUMAN_REVIEWED_NOT_TRAINING_READY"
    ):
        raise ValueError("Missing or mismatched human approval of ConceptNet AI v2 and focus decisions")
    proposals = _csv(proposal_bytes, FIELDS)
    focus = _csv(focus_bytes, FIELDS)
    selections = _csv(selection_bytes, SELECTION_FIELDS)
    with (source_dir / "review.csv").open(encoding="utf-8", newline="") as stream:
        originals = list(csv.DictReader(stream))
    focus_ids = [int(row["class_id"]) for row in focus]
    if (
        len(proposals) != 150
        or len(originals) != 150
        or len(focus_ids) != 16
        or focus_ids != sorted(set(focus_ids))
        or {int(row["class_id"]) for row in proposals if row["proposed_decision"] != "accept"} != set(focus_ids)
        or sum(row["proposed_decision"] == "accept" for row in proposals) != 134
        or [int(row["class_id"]) for row in selections] != accepted_ids
        or not set(accepted_ids).issubset(focus_ids)
    ):
        raise ValueError("AI proposal, focus list, or direct selection IDs differ from the approved 134+16 scope")
    selection_by_id = {int(row["class_id"]): row for row in selections}
    classes, _, entities, relations = _source(kg_dir)
    _, all_assertions = _evidence(kg_dir, classes, entities, relations)
    rows = []
    direct_rows = []
    for i, (original, proposal) in enumerate(zip(originals, proposals, strict=True), start=1):
        if any(original[key] != proposal[key] for key in ("class_id", "label", "conceptnet_uri_candidates")):
            raise ValueError(f"AI proposal/class candidate mismatch at {i}")
        if i in focus_ids:
            if i in selection_by_id:
                selected = selection_by_id[i]
                uri = selected["selected_uri"]
                assertion = selected["evidence_reference"]
                if (
                    uri not in original["conceptnet_uri_candidates"].split(";")
                    or assertion not in all_assertions[i, uri]
                    or not selected["rationale"].strip()
                ):
                    raise ValueError(f"Invalid direct accepted URI/assertion/rationale at class {i}")
                decision, rationale, evidence = "accept", selected["rationale"], assertion
            else:
                uri = ""
                decision = "reject"
                evidence = proposal["evidence_reference"]
                rationale = (
                    f"Direct decision by {approver}: reject the listed candidate URI(s); "
                    f"earlier AI concern: {proposal['ai_rationale']}"
                )
            direct_rows.append(
                {
                    "class_id": i,
                    "label": original["label"],
                    "decision": decision,
                    "selected_uri": uri,
                    "reviewer": approver,
                    "evidence_reference": evidence,
                    "rationale": rationale,
                }
            )
        else:
            decision = proposal["proposed_decision"]
            uri = proposal["proposed_uri"]
            evidence = proposal["evidence_reference"]
            rationale = (
                f"Batch-approved AI proposal {sha256(proposal_bytes)[:12]} by {approver}; {proposal['ai_rationale']}"
            )
        rows.append(
            {
                **{key: original[key] for key in REVIEW_FIELDS if key in original},
                "decision": decision,
                "selected_uri": uri,
                "reviewer": approver,
                "evidence_reference": evidence,
                "rationale": rationale,
            }
        )
    direct_bytes = _csv_bytes(direct_rows, DIRECT_FIELDS)
    provenance = json.loads((source_dir / "provenance.json").read_text(encoding="utf-8"))
    provenance.update(
        purpose="human_approved_ai_proposal_with_direct_focus_decisions",
        review_basis="human_mixed_direct_and_ai_batch",
        approved_by=approver,
        human_direct_class_count=16,
        human_batch_approved_class_count=134,
        approval_sha256=sha256(approval_bytes),
        proposal_sha256=sha256(proposal_bytes),
        focus_sha256=sha256(focus_bytes),
        selection_sha256=sha256(selection_bytes),
        direct_decisions_sha256=sha256(direct_bytes),
    )
    outputs = {
        "review.csv": _csv_bytes(rows, REVIEW_FIELDS),
        "provenance.json": (json.dumps(provenance, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
        "approval.json": approval_bytes,
        "selection.csv": selection_bytes,
        "direct_decisions.csv": direct_bytes,
        "review_suggestions.ai-draft.csv": proposal_bytes,
        "review_focus.ai-draft.csv": focus_bytes,
        "review_suggestions.ai-draft.json": (source_dir / "review_suggestions.ai-draft.json").read_bytes(),
        "evidence_samples.csv": (source_dir / "evidence_samples.csv").read_bytes(),
    }
    with tempfile.TemporaryDirectory() as dirname:
        temporary = Path(dirname)
        for name, data in outputs.items():
            (temporary / name).write_bytes(data)
        counts = check(kg_dir, temporary, require_complete=True)
        if counts != {"accept": 138, "reject": 12, "defer": 0, "pending": 0}:
            raise ValueError(f"Unexpected final review counts: {counts}")
    return outputs, provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kg-dir", type=Path, default=Path("data/knowledge/conceptnet-b2-v1"))
    parser.add_argument("--source-dir", type=Path, default=Path("data/knowledge/conceptnet-review-ai-v2"))
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/knowledge/conceptnet-review-toan-v3"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    outputs, provenance = apply_review(args.kg_dir, args.source_dir, args.approval, args.selection)
    write_outputs(args.output_dir, outputs, dry_run=args.dry_run)
    print(json.dumps({"review_basis": provenance["review_basis"], "counts": {"accept": 138, "reject": 12}}, indent=2))


if __name__ == "__main__":
    main()
