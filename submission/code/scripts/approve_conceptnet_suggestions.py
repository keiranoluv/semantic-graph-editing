"""Apply an explicit human batch approval to a hash-pinned AI draft.

This script never creates an approval declaration. The named person must read
the complete proposal and supply the declaration as a separate input.
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
from scripts.prepare_conceptnet_review import REVIEW_FIELDS, check
from scripts.suggest_conceptnet_review import FIELDS


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def apply_approval(kg_dir: Path, source_dir: Path, approval_file: Path) -> tuple[dict[str, bytes], dict]:
    if check(kg_dir, source_dir) != {"accept": 0, "reject": 0, "defer": 0, "pending": 150}:
        raise ValueError("Expected the untouched ConceptNet review worksheet")
    proposal_bytes = (source_dir / "review_suggestions.ai-draft.csv").read_bytes()
    focus_bytes = (source_dir / "review_focus.ai-draft.csv").read_bytes()
    proposal_info = json.loads((source_dir / "review_suggestions.ai-draft.json").read_text(encoding="utf-8"))
    approval_bytes = approval_file.read_bytes()
    approval = json.loads(approval_bytes)
    approver = approval.get("approver")
    if (
        approval.get("schema_version") != 1
        or approval.get("approval_scope") != "all_150_conceptnet_ai_proposals"
        or not isinstance(approver, str)
        or not approver.strip()
        or approver.strip().casefold() in {"ai", "openai", "chatgpt", "assistant", "llm"}
        or approver != proposal_info.get("requested_human_approver")
        or approval.get("proposal_sha256") != _sha256(proposal_bytes)
        or approval.get("focus_sha256") != _sha256(focus_bytes)
        or not isinstance(approval.get("statement"), str)
        or len(approval["statement"].strip()) < 20
        or not isinstance(approval.get("approved_at_utc"), str)
        or not re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", approval["approved_at_utc"])
    ):
        raise ValueError("Missing or mismatched explicit human batch approval")
    if proposal_info.get("status") != "AI_PROVISIONAL_NOT_HUMAN_REVIEWED_NOT_TRAINING_READY":
        raise ValueError("Expected the unapproved AI proposal provenance")
    with (source_dir / "review.csv").open(encoding="utf-8", newline="") as stream:
        originals = list(csv.DictReader(stream))
    proposal_reader = csv.DictReader(io.StringIO(proposal_bytes.decode("utf-8")))
    if proposal_reader.fieldnames != list(FIELDS):
        raise ValueError("AI proposal columns changed")
    proposals = list(proposal_reader)
    if len(originals) != 150 or len(proposals) != 150:
        raise ValueError("Expected 150 original and proposed rows")
    output_stream = io.StringIO(newline="")
    writer = csv.DictWriter(output_stream, fieldnames=REVIEW_FIELDS, lineterminator="\n")
    writer.writeheader()
    for original, proposal in zip(originals, proposals, strict=True):
        if any(original[key] != proposal[key] for key in ("class_id", "label", "conceptnet_uri_candidates")):
            raise ValueError("AI proposal/class candidate mismatch")
        decision = proposal["proposed_decision"]
        uri = proposal["proposed_uri"]
        if decision not in ("accept", "reject", "defer") or (decision == "accept") != bool(uri):
            raise ValueError(f"Invalid AI proposal at class {original['class_id']}")
        writer.writerow(
            {
                **{key: original[key] for key in REVIEW_FIELDS if key in original},
                "decision": decision,
                "selected_uri": uri,
                "reviewer": approver,
                "evidence_reference": proposal["evidence_reference"],
                "rationale": (
                    f"Batch-approved AI proposal {_sha256(proposal_bytes)[:12]} by {approver}; "
                    + proposal["ai_rationale"]
                ),
            }
        )
    provenance = json.loads((source_dir / "provenance.json").read_text(encoding="utf-8"))
    provenance.update(
        purpose="human_batch_approved_ai_class_link_decisions",
        review_basis="human_batch_approved_ai_proposal",
        approved_by=approver,
        approval_sha256=_sha256(approval_bytes),
        proposal_sha256=_sha256(proposal_bytes),
        focus_sha256=_sha256(focus_bytes),
    )
    outputs = {
        "review.csv": output_stream.getvalue().encode("utf-8"),
        "provenance.json": (json.dumps(provenance, indent=2) + "\n").encode("utf-8"),
        "approval.json": approval_bytes,
        "review_suggestions.ai-draft.csv": proposal_bytes,
        "review_focus.ai-draft.csv": focus_bytes,
        "review_suggestions.ai-draft.json": (source_dir / "review_suggestions.ai-draft.json").read_bytes(),
        "evidence_samples.csv": (source_dir / "evidence_samples.csv").read_bytes(),
    }
    with tempfile.TemporaryDirectory() as dirname:
        temporary = Path(dirname)
        for name, data in outputs.items():
            (temporary / name).write_bytes(data)
        check(kg_dir, temporary, require_complete=True)
    return outputs, provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--kg-dir", type=Path, default=Path("data/knowledge/conceptnet-b2-v1"))
    parser.add_argument("--source-dir", type=Path, default=Path("data/knowledge/conceptnet-review-ai-v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/knowledge/conceptnet-review-approved-ai-v2"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    outputs, provenance = apply_approval(args.kg_dir, args.source_dir, args.approval)
    write_outputs(args.output_dir, outputs, dry_run=args.dry_run)
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
