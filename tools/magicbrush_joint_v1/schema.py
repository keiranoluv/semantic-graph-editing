"""Strict observation schema, deliberately separate from the canonical graph schema."""

import json
import math

VERSION = "magicbrush_joint_observations_v1"


def obj(properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def arr(items, maximum=120):
    return {"type": "array", "items": items, "maxItems": maximum}


def enum(*values):
    return {"type": "string", "enum": list(values)}


TEXT = {"type": "string", "minLength": 1, "maxLength": 1600}
ID = {"type": "string", "pattern": "^[A-Za-z][A-Za-z0-9_]{0,39}$"}
NULL_ID = {"anyOf": [ID, {"type": "null"}]}
ENTITY = obj(
    {"id": ID, "label": {**TEXT, "maxLength": 100}, "attributes": arr({**TEXT, "maxLength": 100}, 8), "evidence": TEXT}
)
RELATION = obj({"id": ID, "subject": ID, "predicate": {**TEXT, "maxLength": 100}, "object": ID, "evidence": TEXT})
CORRESPONDENCE = obj(
    {"source_id": NULL_ID, "target_id": NULL_ID, "evidence": TEXT, "certainty": enum("clear", "uncertain")}
)
CHANGE = obj(
    {
        "entity_type": enum("node", "edge"),
        "source_id": NULL_ID,
        "target_id": NULL_ID,
        "change": enum("addition", "removal", "replacement"),
        "source_evidence": TEXT,
        "target_evidence": TEXT,
        "unintended": {"type": "boolean"},
    }
)
SCHEMA = obj(
    {
        "schema_version": enum(VERSION),
        "instruction_fulfilled": enum("yes", "no", "partial", "uncertain"),
        "annotation_status": enum("supported", "ambiguous", "unsupported", "unsuccessful"),
        "source_entities": arr(ENTITY, 20),
        "target_entities": arr(ENTITY, 20),
        "source_relations": arr(RELATION, 60),
        "target_relations": arr(RELATION, 60),
        "entity_correspondences": arr(CORRESPONDENCE, 40),
        "relation_correspondences": arr(CORRESPONDENCE, 120),
        "observed_changes": arr(CHANGE),
        "unchanged_entities": arr(ID, 20),
        "uncertainties": arr(TEXT, 30),
        "explanation": TEXT,
        "model_confidence": {"type": "number", "minimum": 0, "maximum": 1},
    }
)


def validate(value, schema=SCHEMA, path="$"):
    """Small strict JSON-schema subset; no extra runtime dependencies or coercions.

    The identical schema is sent to vLLM for constrained decoding. Semantic
    correspondence coverage and visual-evidence policy belong to the compiler.
    """
    if "anyOf" in schema:
        for option in schema["anyOf"]:
            try:
                validate(value, option, path)
                return value
            except ValueError:
                pass
        raise ValueError(f"schema:{path}: no matching type")
    kind = schema["type"]
    expected = {"object": dict, "array": list, "string": str, "null": type(None), "boolean": bool}
    valid_type = type(value) in (int, float) if kind == "number" else type(value) is expected[kind]
    if not valid_type:
        raise ValueError(f"schema:{path}: expected {kind}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"schema:{path}: invalid enum")
    if kind == "object":
        if set(value) != set(schema["properties"]):
            raise ValueError(f"schema:{path}: missing/extra fields")
        for key, item in value.items():
            validate(item, schema["properties"][key], f"{path}.{key}")
    elif kind == "array":
        if len(value) > schema["maxItems"]:
            raise ValueError(f"schema:{path}: excessive items")
        for i, item in enumerate(value):
            validate(item, schema["items"], f"{path}[{i}]")
    elif kind == "string":
        import re

        if not value.strip() or not schema.get("minLength", 0) <= len(value) <= schema.get("maxLength", 100000):
            raise ValueError(f"schema:{path}: invalid text length")
        if "pattern" in schema and not re.fullmatch(schema["pattern"], value):
            raise ValueError(f"schema:{path}: invalid ID")
    elif kind == "number" and (not math.isfinite(value) or not schema["minimum"] <= value <= schema["maximum"]):
        raise ValueError(f"schema:{path}: invalid number")
    return value


def parse(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("schema:duplicate JSON key")
            result[key] = value
        return result

    return validate(json.loads(raw, object_pairs_hook=unique))
