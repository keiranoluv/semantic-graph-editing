"""Deterministic grouped splits, including multi-turn editing sessions."""

import hashlib


def grouped_split(records, *, group_key="source_group_id", seed=42, val_fraction=0.15, test_fraction=0.15):
    """All turns/variants of an original source image stay together.

    Fractions apply to groups, not records. Hash ordering is independent of input
    row order. This helper is for a project subset, not an official benchmark split."""
    if not 0 <= val_fraction < 1 or not 0 <= test_fraction < 1 or val_fraction + test_fraction >= 1:
        raise ValueError("Invalid split fractions")
    records = list(records)
    ids = [r["id"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate example IDs")
    if any(not isinstance(r[group_key], str) or not r[group_key] for r in records):
        raise ValueError("Every record requires a nonempty source group ID")
    groups = sorted(
        {r[group_key] for r in records},
        key=lambda g: (hashlib.sha256(f"{seed}:{g}".encode()).hexdigest(), g),
    )
    nval, ntest = int(len(groups) * val_fraction), int(len(groups) * test_fraction)
    membership = {g: "val" if i < nval else "test" if i < nval + ntest else "train" for i, g in enumerate(groups)}
    result = {k: [] for k in ("train", "val", "test")}
    for r in sorted(records, key=lambda r: r["id"]):
        result[membership[r[group_key]]].append(r["id"])
    return result
