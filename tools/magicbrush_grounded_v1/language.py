"""Conservative, auditable instruction grammar; unknown meaning is deferred."""

import re
from functools import lru_cache

from tools.prepare_magicbrush import ROOT

COLORS = ("black", "white", "gray", "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown")
TAGGER = ROOT / "data/external/anchor-nltk-v1"
TAGGER_FILES = {
    "weights": "789e8e35f6fac656d9b7eccea29ba4e8a137cbb5b525dd31f39e4291a9e80832",
    "tagdict": "4c713731cb06727736962bc8cd94ad919217a040e76691059af99bc0c2c9c246",
    "classes": "d152ffa69d45b8357276341a3d592c6c025911d52ec0d9068d4f382560dd48b6",
}


@lru_cache(maxsize=1)
def tagger():
    import nltk

    from src.knowledge.transe.core import sha256

    for suffix, expected in TAGGER_FILES.items():
        path = TAGGER / f"taggers/averaged_perceptron_tagger_eng/averaged_perceptron_tagger_eng.{suffix}.json"
        if sha256(path) != expected:
            raise ValueError("Pinned CPU POS tagger differs; run --stage prepare first")
    if str(TAGGER) not in nltk.data.path:
        nltk.data.path.insert(0, str(TAGGER))
    return nltk.tag.PerceptronTagger()


def clean(text):
    return " ".join(re.sub(r"[^a-z ]", " ", text.casefold()).split())


def phrase(text):
    """Only singular noun compounds + optional color. No silent modifier loss."""
    text = re.sub(r"^(?:the|a|an|this|that|entire|whole)\s+", "", clean(text))
    text = re.sub(r"^(?:entire|whole)\s+", "", text)
    words = text.split()
    colors = [w for w in words if w in COLORS]
    words = [w for w in words if w not in COLORS]
    if not 1 <= len(words) <= 3 or len(colors) > 1:
        return None
    tagged = tagger().tag(["the", *words])[1:]
    if any(tag not in ("NN", "NNP") for _, tag in tagged):
        return None
    if any(
        w
        in {
            "image",
            "photo",
            "picture",
            "background",
            "scene",
            "color",
            "colour",
            "one",
            "thing",
            "it",
            "s",
            "his",
            "her",
            "their",
            "its",
        }
        for w in words
    ):
        return None
    return {"label": " ".join(words), "color": colors[0] if colors else None}


def scope(instruction):
    text = clean(instruction).replace("grey", "gray")
    text = re.sub(r"^(?:please |let s )", "", text)
    reject = {"supported": False, "reason": "unsupported_or_ambiguous_instruction"}
    if re.search(
        r"\b(and|or|both|all|two|three|second|some|not|never|without|style|cartoon|painting|text|words|font|blur)\b",
        text,
    ):
        return {**reject, "reason": "compound_count_negative_or_nonlocal_edit"}
    match = re.fullmatch(r"(?:replace|swap) (.+?) (?:with|for) (.+)", text)
    if not match:
        match = re.fullmatch(r"(?:change|turn) (.+?) (?:to|into) (.+)", text)
    if match:
        before, after = phrase(match[1]), phrase(match[2])
        if before and after and before["label"] != after["label"]:
            return {"supported": True, "kind": "replace", "before": before, "after": after}
    match = re.fullmatch(
        r"(?:make|paint|color|colour|turn|change|let) (.+?) (?:to |be |become )?(" + "|".join(COLORS) + r")", text
    )
    if match and (entity := phrase(match[1])):
        return {"supported": True, "kind": "color", "before": entity, "after": {**entity, "color": match[2]}}
    match = re.fullmatch(r"(?:add|insert|put) (.+?)(?: (?:in|to|on) the (?:image|picture|photo))?", text)
    if match and (entity := phrase(match[1])):
        return {"supported": True, "kind": "add", "after": entity}
    match = re.fullmatch(r"(?:remove|delete|erase|get rid of) (.+)", text)
    if match and (entity := phrase(match[1])):
        return {"supported": True, "kind": "delete", "before": entity}
    match = re.fullmatch(r"move (.+?) (above|below|to the left of|to the right of) (.+)", text)
    if match:
        entity, context = phrase(match[1]), phrase(match[3])
        if entity and context and entity["label"] != context["label"]:
            direction = {"to the left of": "left_of", "to the right of": "right_of"}.get(match[2], match[2])
            return {
                "supported": True,
                "kind": "move",
                "before": entity,
                "after": entity,
                "context": context,
                "direction": direction,
            }
    return reject


def mentions(intent):
    return sorted({intent[k]["label"] for k in ("before", "after", "context") if k in intent})


def caption_nouns(caption):
    """Source caption proposes noun compounds; boxes must independently support them."""
    tagged = tagger().tag(clean(caption).split())
    groups, run = [], []
    for word, tag in [*tagged, ("", "END")]:
        if tag in ("NN", "NNP"):
            run.append(word)
        else:
            if run and len(run) <= 3 and (parsed := phrase(" ".join(run))):
                groups.append(parsed["label"])
            run = []
    return sorted(set(groups))
