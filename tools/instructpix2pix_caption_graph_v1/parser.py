"""Small fully consumed grammar backed by the existing hash-pinned NLTK POS tagger.

No partial parse is accepted. These are caption semantics, never visual truth.
"""

import re
from dataclasses import dataclass

from src.editing.operations import Edge, EditGraph, Node
from tools.magicbrush_grounded_v1.language import COLORS, tagger

ATTRIBUTES = frozenset((*COLORS, "small", "large", "big", "giant", "tiny", "tall", "short", "wooden", "metal", "shiny"))
STYLE = re.compile(
    r"\b(?:style|painting|drawing|cartoon|anime|art|photo|photograph|image|picture|render|portrait|illustration|sketch|watercolor|digital|abstract|surreal|cinematic|background|scene)\b"
)
PRONOUN = re.compile(r"\b(?:it|its|he|she|his|her|they|them|their|this|that|these|those|someone|something)\b")
RELATIONS = {
    "to the left of": "left_of",
    "to the right of": "right_of",
    "in front of": "in_front_of",
    "next to": "beside",
    "on top of": "on",
    "underneath": "under",
    "beside": "beside",
    "behind": "behind",
    "above": "above",
    "below": "below",
    "inside": "in",
    "under": "under",
    "near": "near",
    "on": "on",
    "in": "in",
}
RELATION_PATTERN = "|".join(re.escape(x) for x in sorted(RELATIONS, key=lambda x: (-len(x), x)))
# Explicit compounds prevent interpreting an unknown adjective as a noun modifier.
COMPOUNDS = frozenset(
    {
        "living room",
        "dining room",
        "coffee table",
        "fire hydrant",
        "traffic light",
        "polar bear",
        "palm tree",
        "tennis racket",
        "baseball bat",
        "wine glass",
        "park bench",
        "hot dog",
    }
)
NONOBJECTS = frozenset(
    {
        "color",
        "colour",
        "thing",
        "one",
        "view",
        "sunrise",
        "sunset",
        "night",
        "day",
        "weather",
        "light",
        "reflection",
        "water",
        "sky",
        "text",
        "letter",
        "word",
        "personality",
    }
)


@dataclass(frozen=True)
class Parsed:
    graph: EditGraph
    evidence: dict


def clean(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty_or_nonstring_caption")
    text = " ".join(text.casefold().replace("grey", "gray").split()).rstrip(".")
    if not re.fullmatch(r"[a-z -]+", text) or len(text.split()) > 40:
        raise ValueError("unsupported_punctuation_length_or_number")
    if STYLE.search(text):
        raise ValueError("style_global_or_metacaption")
    return text


def noun_phrase(text):
    """Accept singular common objects and known attributes, without dropping words."""
    text = re.sub(r"^(?:a|an|the) ", "", text.strip())
    if PRONOUN.search(text):
        raise ValueError("unresolved_pronoun")
    words = text.split()
    attributes = []
    while words and words[0] in ATTRIBUTES:
        attributes.append(words.pop(0))
    if not words or len(words) > 2 or len(attributes) != len(set(attributes)):
        raise ValueError("unsupported_noun_phrase")
    label = " ".join(words)
    if any(w in NONOBJECTS for w in words) or (len(words) > 1 and label not in COMPOUNDS):
        raise ValueError("unsupported_object_or_compound")
    tags = tagger().tag(["a", *words])[1:]
    if any(tag != "NN" for _, tag in tags) and label not in COMPOUNDS:
        raise ValueError("nonsingular_or_noncommon_object")
    if len([a for a in attributes if a in COLORS]) > 1:
        raise ValueError("ambiguous_multiple_colors")
    return label, tuple(sorted(attributes)), {"phrase": text, "pos": tags}


def parse(caption):
    text = clean(caption)
    nodes, edges, phrases, roles = [], [], [], {}

    def entity(phrase, role):
        label, attributes, evidence = noun_phrase(phrase)
        if label in {n.label for n in nodes}:
            raise ValueError("multiple_indistinguishable_instances")
        identifier = f"n{len(nodes)+1:03d}"
        nodes.append(Node(identifier, label, attributes))
        phrases.append(evidence)
        roles[identifier] = role
        return identifier

    # The only supported anaphora: a unique head object, with one contained
    # clause whose "it" must refer to that head. Other pronouns are rejected.
    nested = re.fullmatch(r"(.+?) with (.+?) on it(?: in (.+))?", text)
    posture = None
    if nested:
        head = entity(nested[1], "head")
        item = entity(nested[2], "subject")
        edges.append(Edge("e001", item, "on", head))
        if nested[3]:
            room = entity(nested[3], "object")
            edges.append(Edge("e002", head, "in", room))
    else:
        if PRONOUN.search(text):
            raise ValueError("unresolved_pronoun")
        # Static posture is retained as evidence; changes are rejected by compiler.
        text, count = re.subn(
            r"\s+(?:is )?(sitting|standing|parked)(?= (?:" + RELATION_PATTERN + r") )", r" __\1__", text
        )
        match_posture = re.search(r" __(sitting|standing|parked)__", text)
        if count > 1:
            raise ValueError("unsupported_multiple_actions")
        if match_posture:
            posture = match_posture[1]
            text = text.replace(match_posture[0], "")
        text = re.sub(r"\s+is(?= (?:" + RELATION_PATTERN + r") )", "", text)
        parts = re.split(r" (" + RELATION_PATTERN + r") ", text)
        if len(parts) == 3:
            subject = entity(parts[0], "subject")
            obj = entity(parts[2], "object")
            edges.append(Edge("e001", subject, RELATIONS[parts[1]], obj))
        elif len(parts) == 1:
            copula = re.fullmatch(r"(.+?) is (" + "|".join(sorted(ATTRIBUTES)) + r")", text)
            if copula:
                label, attrs, _ = noun_phrase(copula[1])
                if attrs:
                    raise ValueError("ambiguous_predicative_attribute")
                entity(f"{copula[2]} {label}", "head")
            else:
                # Conjunction only supports a flat list, with no relation scope.
                for phrase in text.split(" and "):
                    entity(phrase, "head")
        else:
            raise ValueError("ambiguous_relation_attachment")
    return Parsed(
        EditGraph(tuple(nodes), tuple(edges)),
        {
            "grammar": "fully_consumed_v1",
            "phrases": phrases,
            "roles": roles,
            "posture": posture,
            "resolved_anaphora": bool(nested),
        },
    )
