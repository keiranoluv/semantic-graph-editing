"""Caption-only construction; instruction checks the diff but never supplies it."""

from src.editing.operations import diff_aligned_graphs
from tools.instructpix2pix_caption_graph_v1.alignment import align
from tools.instructpix2pix_caption_graph_v1.instruction_check import consistent, intent
from tools.instructpix2pix_caption_graph_v1.parser import parse
from tools.instructpix2pix_caption_graph_v1.validator import validate


def compile_record(record):
    result = {
        "id": record["id"],
        "status": "rejected",
        "eligible_for_training": False,
        "problems": [],
        "human_review_status": "pending",
        "visual_verification_status": "unverified",
        "stages": {
            k: None
            for k in (
                "fields",
                "source_parse",
                "target_parse",
                "caption_parsing",
                "alignment",
                "instruction_consistency",
                "structural_validity",
            )
        },
        "evidence": {},
    }
    stages, evidence = result["stages"], result["evidence"]
    required = ("original_prompt", "edited_prompt", "edit_prompt")
    stages["fields"] = all(isinstance(record.get(k), str) and record[k].strip() for k in required)
    if not stages["fields"]:
        result["problems"].append("missing_or_invalid_caption_instruction_fields")
        return result
    parsed = {}
    for role, field in (("source", "original_prompt"), ("target", "edited_prompt")):
        try:
            parsed[role] = parse(record[field])
            stages[f"{role}_parse"] = True
            evidence[f"{role}_parse"] = parsed[role].evidence
            result[f"{role}_graph"] = parsed[role].graph.to_dict()
        except ValueError as error:
            stages[f"{role}_parse"] = False
            result["problems"].append(f"{role}_parse:{error}")
    stages["caption_parsing"] = all(stages[f"{r}_parse"] for r in ("source", "target"))
    if not stages["caption_parsing"]:
        return result
    try:
        request = intent(record["edit_prompt"])
        evidence["instruction_intent"] = request
    except ValueError as error:
        stages["instruction_consistency"] = False
        result["problems"].append(f"instruction:{error}")
        return result
    try:
        source, target, evidence["alignment"] = align(parsed["source"], parsed["target"], request)
        stages["alignment"] = True
    except ValueError as error:
        stages["alignment"] = False
        result["problems"].append(f"alignment:{error}")
        return result
    operations = diff_aligned_graphs(source, target)
    result.update(
        source_graph=source.to_dict(), target_graph=target.to_dict(), operations=[o.to_dict() for o in operations]
    )
    posture_equal = parsed["source"].evidence["posture"] == parsed["target"].evidence["posture"]
    stages["instruction_consistency"] = (
        posture_equal and bool(operations) and consistent(source, target, operations, request)
    )
    if not stages["instruction_consistency"]:
        result["problems"].append("instruction_caption_delta_disagreement_or_unrepresented_change")
    try:
        source, target, operations = validate(source, target, operations)
        stages["structural_validity"] = True
        result.update(
            source_graph=source.to_dict(), target_graph=target.to_dict(), operations=[o.to_dict() for o in operations]
        )
    except ValueError as error:
        stages["structural_validity"] = False
        result["problems"].append(f"structure:{error}")
    if not result["problems"]:
        result.update(status="candidate", eligible_for_training=True)
    return result
