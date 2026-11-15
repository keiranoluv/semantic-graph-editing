"""Strict region-identity parsing and loss-aware EditGraph normalization on CPU."""

from __future__ import annotations

import re

from src.editing.operations import Edge, EditGraph, Node
from tools.level3_image_graph_v1.core import digest, validate_box

from . import VERSION

# Only consecutive leading modifiers belong unambiguously to the described head.
# Do not extract "red" from "a man wearing a red shirt" onto the man.
MODIFIERS = frozenset(
    (
        "black white gray grey red green blue yellow orange brown pink purple wooden metal metallic "
        "plastic glass stone brick concrete fabric striped spotted round rectangular small large"
    ).split()
)


class ParseError(ValueError):
    pass


def parse_robin(text, region_count):
    """Native format: Objects: regionN: description; Relations: regionN: regionM predicate.

    The official parser drops region numbers and silently skips malformed edges.
    Here region identity is retained, missing objects/unknown lines are errors,
    and every raw statement is accounted for. No eval/JSON repair/LLM rewriting.
    """
    if not isinstance(text, str) or len(text) > 200_000 or not 1 <= region_count <= 99:
        raise ParseError("invalid_native_input")
    pattern = r"\A\s*Objects:\s*(.*?)\s*Relations:\s*(.*?)\s*\Z"
    match = re.fullmatch(pattern, text, flags=re.S)
    if not match:
        raise ParseError("missing_native_sections")
    object_part, relation_part = match.groups()
    objects = {}
    object_matches = list(re.finditer(r"(?:\A|\n)\s*region(\d+):\s*(.*?)(?=\n\s*region\d+:|\Z)", object_part, re.S))
    consumed = "".join(m.group(0) for m in object_matches)
    if consumed.strip() != object_part.strip():
        raise ParseError("unparsed_object_text")
    for item in object_matches:
        region_id = int(item.group(1))
        description = item.group(2).strip()
        if region_id in objects or not 1 <= region_id <= region_count or not description:
            raise ParseError("duplicate_unknown_or_empty_object")
        if "\n" in description or re.search(r"region\d+:", description):
            raise ParseError("ambiguous_multiline_description")
        objects[region_id] = description
    if set(objects) != set(range(1, region_count + 1)):
        raise ParseError("missing_region_descriptions")
    relations, duplicate_rows, seen = [], [], set()
    if relation_part.strip().casefold() not in {"", "none", "none."}:
        for line in relation_part.splitlines():
            if not line.strip():
                continue
            row = re.fullmatch(r"\s*region(\d+):\s*(.*?)\s*", line)
            if not row or int(row.group(1)) not in objects:
                raise ParseError("invalid_relation_source")
            source = int(row.group(1))
            if row.group(2).casefold() in {"none", "none.", ""}:
                continue
            # A comma inside a predicate is not a separator unless followed by regionN.
            for fragment in re.split(r",\s*(?=region\d+\b)", row.group(2)):
                edge = re.fullmatch(r"region(\d+)\s+([^\n]+)", fragment.strip())
                if not edge or int(edge.group(1)) not in objects:
                    raise ParseError("invalid_relation_target")
                target, predicate = int(edge.group(1)), edge.group(2).strip()
                if not predicate or re.search(r"\bregion\d+\b", predicate):
                    raise ParseError("ambiguous_relation_predicate")
                triple = (source, predicate, target)
                if triple in seen:
                    duplicate_rows.append({"source": source, "target": target, "predicate": predicate, "raw": fragment})
                else:
                    seen.add(triple)
                    relations.append(triple)
    return objects, relations, duplicate_rows


def description_attributes(description):
    """Conservative textual projection; not a separate visual attribute predictor."""
    prefix = re.sub(r"^(?:a|an|the)\s+", "", description.strip(), flags=re.I)
    matches = list(re.finditer(r"[A-Za-z]+", prefix))
    attributes = []
    end = 0
    for i, match in enumerate(matches):
        token = match.group().lower()
        if prefix[end : match.start()].strip() or token not in MODIFIERS:
            break
        # Leave an eventual head noun: "a glass" alone is not a glass attribute.
        if i + 1 == len(matches):
            break
        attributes.append(token)
        end = match.end()
    return sorted(set(attributes))


def validate_regions(regions, width, height):
    if not regions or len(regions) > 99:
        raise ValueError("expected_1_to_99_regions")
    if [r["region_id"] for r in regions] != list(range(1, len(regions) + 1)):
        raise ValueError("region_ids_must_be_ordered_and_contiguous")
    for region in regions:
        validate_box(region["bbox_xyxy"], width, height)
        if region["segmentation"]["size"] != [height, width]:
            raise ValueError("mask_image_size_mismatch")


def robin_record(row, regions, text, provenance, generation):
    validate_regions(regions, row["width"], row["height"])
    if generation.get("hit_token_limit"):
        raise ParseError("generation_hit_token_limit")
    objects, triples, duplicates = parse_robin(text, len(regions))
    nodes, edges, grounding = [], [], {"nodes": {}, "edges": {}}
    for region in regions:
        rid = region["region_id"]
        identifier = f"n{rid:03d}"
        description = objects[rid]
        attrs = description_attributes(description)
        # The native object label is a region description, not a categorical noun.
        # Preserve it verbatim. Human review must compare meaning, not string equality.
        nodes.append(Node(identifier, description, tuple(attrs)))
        grounding["nodes"][identifier] = {
            "bbox_xyxy": region["bbox_xyxy"],
            "confidence": None,
            "confidence_kind": "not_provided_by_robin",
            "origin": "robin_region_description",
            "raw_detection_ids": [f"region{rid}"],
            "native_description": description,
            "label_kind": "region_description",
            "region": region,
            "grounding_origin": "sam_input_region_not_robin_predicted_box",
            "attributes": [
                {
                    "label": a,
                    "confidence": None,
                    "confidence_kind": "not_provided",
                    "origin": "explicit_leading_modifier_projection",
                    "evidence": description,
                }
                for a in attrs
            ],
        }
    for i, (source, predicate, target) in enumerate(triples, 1):
        identifier = f"e{i:03d}"
        edges.append(Edge(identifier, f"n{source:03d}", predicate, f"n{target:03d}"))
        grounding["edges"][identifier] = {
            "confidence": None,
            "confidence_kind": "not_provided_by_robin",
            "origin": "robin_native_directed_relation",
            "raw_relation_ids": [i],
            "native_triplet": [source, target, predicate],
        }
    graph = EditGraph(tuple(nodes), tuple(edges))
    record = {
        "schema": VERSION,
        "model": "robin-sam",
        "candidate_id": row["image_id"],
        "image_id": row["image_id"],
        "image": row,
        "source_sha256": row["sha256"],
        "source_graph": graph.to_dict(),
        "grounding": grounding,
        "provenance": provenance,
        "native_output": {
            "text": text,
            "generation": generation,
            "duplicate_relations": duplicates,
            "region_count": len(regions),
        },
        "annotation_status": "needs_review",
        "human_reviewed": False,
        "attribute_coverage": "partial_explicit_prefix_only; other properties remain in native descriptions",
        "native_text_sha256": digest(text),
    }
    return record
