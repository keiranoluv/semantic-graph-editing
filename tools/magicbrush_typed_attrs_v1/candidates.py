"""Attribute candidates from instruction spans and source fields only."""
from tools.magicbrush_easy_copy_v1.candidates import candidates as name_candidates

from .schema import PARTS, attribute, color_atoms

MAX_ATTRIBUTES = 192
CONTENT_CUES = {
    "screen",
    "monitor",
    "display",
    "picture",
    "image",
    "photo",
    "photograph",
    "painting",
    "print",
    "graphic",
    "wallpaper",
}


def attribute_candidates(source, instruction):
    spans = name_candidates(instruction)
    fields = {a.split("=", 1)[0] for n in source.nodes for a in n.attributes if "=" in a}
    tokens = set(instruction.lower().split())
    values = []

    def add(value):
        if value not in values:
            values.append(value)

    # Candidate order independent of targets; all direct atoms have priority.
    for span in spans:
        for atom in attribute(span):
            add(atom)
    for span in spans:
        if tokens & CONTENT_CUES:
            add("content=" + span)
        if tokens & {"text", "lettering", "write", "say", "says", "word"}:
            add("text=" + span.upper())
        if color_atoms(span):
            for field in sorted(fields):
                if field == "color" or field.endswith(".color"):
                    for atom in color_atoms(span, field):
                        add(atom)
            for part in sorted(tokens & PARTS):
                for atom in color_atoms(span, part + ".color"):
                    add(atom)
    return tuple(values[:MAX_ATTRIBUTES])
