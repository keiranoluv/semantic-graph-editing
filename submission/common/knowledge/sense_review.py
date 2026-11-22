"""Reproducible, human-reviewed WordNet sense disambiguation for VG150.

A ranking is a *suggestion*, not a verified link. Only an explicit accept
choice in a review CSV is compiled to a curated WordNet override.
"""


def _validate_mapping(mapping: dict) -> list[dict]:
    """Reject malformed or reordered class IDs before any review work."""
    rows = mapping.get("rows")
    if mapping.get("schema_version") != 1 or mapping.get("source") != "wordnet" or not isinstance(rows, list):
        raise ValueError("Expected Milestone A WordNet mapping with schema_version 1")
    if not rows or rows[0].get("class_id") != 0 or rows[0].get("status") != "background":
        raise ValueError("Mapping must start with background class ID 0")
    for idx, row in enumerate(rows):
        if type(row.get("class_id")) is not int or row["class_id"] != idx:
            raise ValueError("Mapping class IDs must be contiguous and ordered")
        if idx and (
            not isinstance(row.get("label"), str)
            or row.get("status") not in {"linked_auto", "linked_manual", "ambiguous", "unmatched"}
            or not isinstance(row.get("candidates"), list)
        ):
            raise ValueError(f"Invalid mapping entry at ID {idx}")
    return rows
