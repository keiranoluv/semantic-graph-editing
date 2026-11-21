"""Normalized, framework-independent scene graph schemas.

Object IDs are instance identifiers; label IDs are vocabulary indices.
Relationship endpoints reference object IDs, not row positions.
Coordinates use absolute pixel-space xyxy format.
"""

from dataclasses import dataclass, field
from math import isfinite


@dataclass(frozen=True)
class SceneObject:
    """One object instance within an image."""

    object_id: int
    label_id: int
    bbox_xyxy: tuple[float, float, float, float]
    attribute_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class SceneRelation:
    """A directed annotated subject-predicate-object triple."""

    subject_id: int
    object_id: int
    predicate_id: int


@dataclass(frozen=True)
class SceneGraphRecord:
    """Single-image normalized annotation; no PyG dependency."""

    image_id: str
    width: int
    height: int
    objects: tuple[SceneObject, ...] = ()
    relations: tuple[SceneRelation, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        """Check minimal representation invariants (not dataset parsing)."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Image dimensions must be positive")
        ids = [obj.object_id for obj in self.objects]
        if len(ids) != len(set(ids)):
            raise ValueError("Object instance IDs must be unique")
        for obj in self.objects:
            x1, y1, x2, y2 = obj.bbox_xyxy
            if not (0 <= x1 < x2 <= self.width and 0 <= y1 < y2 <= self.height):
                raise ValueError("Invalid xyxy bounding box")
            if not all(isfinite(value) for value in obj.bbox_xyxy):
                raise ValueError("Bounding box coordinates must be finite")
            if obj.label_id < 0:
                raise ValueError("Object label IDs must be nonnegative")
            if any(attribute_id < 0 for attribute_id in obj.attribute_ids):
                raise ValueError("Attribute IDs must be nonnegative")
        known = set(ids)
        for relation in self.relations:
            if relation.subject_id not in known or relation.object_id not in known:
                raise ValueError("Relationship endpoint not in objects")
            if relation.predicate_id <= 0:
                raise ValueError("Annotated predicates must have IDs >= 1")
