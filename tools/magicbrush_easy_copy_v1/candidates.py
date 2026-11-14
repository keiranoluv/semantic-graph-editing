"""Per-example instruction spans; no target, external parser or pooled val lexicon."""
from functools import lru_cache

from tools.magicbrush_lexical_pilot_v1.features import words

# Keep all unigram candidates before longer spans. These are candidate boundaries,
# not labels or target-derived statistics. No noun dictionary restricts copying.
BOUNDARY = set(
    """a an the this that these those some any please can could would will
should may might must you i we it its they them their there be being been is are was
were to into onto from of for at by on in out up down with without and or but instead
add added insert put place remove delete erase change replace turn make let have
get give take rid want need looking look wearing wear holding hold playing play
crossing cross standing stand sitting sit become show showing appear appearing
now also just do does not no as than then""".split()
)
MAX_CANDIDATES = 64


@lru_cache(maxsize=4096)
def candidates(instruction):
    tokens = words(instruction)
    values = []
    for size in (1, 2, 3, 4):
        for start in range(len(tokens) - size + 1):
            span = tokens[start : start + size]
            if any(w in BOUNDARY for w in span):
                continue
            value = " ".join(span)
            if value not in values:
                values.append(value)
    return tuple(values[:MAX_CANDIDATES])
