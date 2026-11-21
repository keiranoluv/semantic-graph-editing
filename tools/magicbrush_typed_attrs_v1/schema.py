"""Conservative typed attributes. Unknown phrases survive as detail=... .

No instruction or target graph is consulted when projecting an individual node.
Known semantic aliases and full-phrase grammars are explicit. Position, polarity,
comparatives and content are preserved; no whitelist-drop fallback exists.
"""
import re
from dataclasses import replace
from functools import lru_cache

from src.editing.operations import EditGraph, apply_operations, diff_aligned_graphs

COLORS = set(
    "red green blue yellow orange purple pink black white gray grey brown blond blonde gold golden silver beige tan turquoise maroon violet multicolored".split()
)
SHADES = set("dark light pale bright deep navy".split())
MATERIALS = set(
    "wood wooden metal metallic leather rubber glass plastic fabric stone concrete brick steel silver".split()
)
STATES = set(
    "open closed upright lying standing sitting bald wet dry empty full broken asleep sleeping plain sparse dense denser taller shorter larger smaller".split()
)
SHAPES = set("round oval rectangular square circular conical triangular".split())
SIZES = set("large small big tiny huge tall short long thin thick wide narrow".split())
PARTS = set(
    "hair shirt coat dress pant pants tie hat frame border header background face neck chest beak nose eye eyes mouth head sleeve sleeves side sides front back top bottom shoulder strap icing seam seams roof handle seat rim".split()
)
CONTENT_NODES = {"screen", "monitor", "display"}
ALIASES = {
    "grey": "gray",
    "blonde": "blond",
    "golden": "gold",
    "wooden": "wood",
    "metallic": "metal",
    "sleeping": "asleep",
}


def color_atoms(text, prefix="color"):
    tokens = text.lower().split()
    if not tokens or any(t not in COLORS | SHADES | {"and", ","} for t in tokens):
        return ()
    # Associate shades with their color; never discard modifiers.
    values, modifiers = [], []
    for t in tokens:
        if t in SHADES:
            modifiers.append(t)
        elif t in COLORS:
            values.append(prefix + "=" + " ".join([*modifiers, ALIASES.get(t, t)]))
            modifiers = []
        elif modifiers:
            return ()
    return tuple(values) if not modifiers else ()


@lru_cache(maxsize=8192)
def attribute(text, node_label=""):
    raw = " ".join(text.strip().split())
    if "=" in raw:
        return (raw,)  # idempotence on this version's typed atoms
    low = raw.lower()
    colors = color_atoms(low)
    if colors:
        return tuple(sorted(set(colors)))
    if low in MATERIALS:
        return ("material=" + ALIASES.get(low, low),)
    if low in STATES:
        return ("state=" + ALIASES.get(low, low),)
    if low in SHAPES:
        return ("shape=" + low,)
    if low in SIZES:
        return ("size=" + low,)
    if low in {
        "left",
        "right",
        "center",
        "upper left",
        "upper right",
        "lower left",
        "lower right",
        "foreground",
        "background",
    }:
        return ("position=" + low,)
    # Scoped color of a part does not become color of the entire object.
    for part in sorted(PARTS, key=lambda s: (-len(s), s)):
        if low.endswith(" " + part):
            cs = color_atoms(low[: -(len(part) + 1)], part + ".color")
            if cs:
                return tuple(sorted(set(cs)))
        if low.startswith(part + " "):
            value = low[len(part) + 1 :]
            if value in STATES or value in {"raised", "resting", "visible", "wide open"}:
                return (part + ".state=" + ALIASES.get(value, value),)
    if low.startswith("facing "):
        return ("direction=" + low[7:],)
    # Literal text retains case, e.g. SALE is not a generic adjective.
    match = re.fullmatch(r"(?:(red|green|blue|yellow|black|white) )?(.+?) (text|lettering)", raw)
    if match and match[2].upper() == match[2]:
        return tuple(sorted(["text=" + match[2]] + (["text.color=" + match[1]] if match[1] else [])))
    content = re.fullmatch(r"(.+?) (image|graphic|print|wallpaper)", low)
    if content:
        value = content[1]
        modifiers = []
        while value.split()[0] in COLORS | SIZES:
            first, _, rest = value.partition(" ")
            if not rest:
                break
            modifiers.append(("content.color=" if first in COLORS else "content.size=") + ALIASES.get(first, first))
            value = rest
        return tuple(sorted(["content=" + value, *modifiers]))
    if node_label.lower() in CONTENT_NODES:
        if low.endswith(" windows"):
            return ("content=" + low[:-8],)
        if low in {"webpage", "desktop"}:
            return ("content=" + low,)
        match = re.fullmatch(r"(.+?) (desktop|webpage)", low)
        if match and color_atoms(match[1], "content.color"):
            return tuple(sorted(("content=" + match[2], *color_atoms(match[1], "content.color"))))
    # General adjective + part phrase, e.g. purple shoulder strap.
    for cut in range(1, len(low.split())):
        prefix = " ".join(low.split()[:cut])
        suffix = " ".join(low.split()[cut:])
        if color_atoms(prefix) and not any(w in suffix.split() for w in ["and", "with", "without", "not"]):
            return tuple(sorted(color_atoms(prefix, suffix.replace(" ", "_") + ".color")))
    return ("detail=" + raw,)  # unrecognized phrases retained by default


def graph(value):
    return EditGraph(
        tuple(
            replace(n, attributes=tuple(sorted({v for a in n.attributes for v in attribute(a, n.label)})))
            for n in value.nodes
        ),
        value.edges,
    )


def project(example):
    source, target = graph(example.source), graph(example.target)
    ops = diff_aligned_graphs(source, target)
    if apply_operations(source, ops) != target:
        raise AssertionError("Typed graph transaction failed")
    return replace(example, source=source, target=target, operations=ops)
