"""Source identity and runtime versions for the graph-edit experiment."""

import importlib.metadata


def source_hashes():
    from common.provenance import source_hashes as current_sources

    return current_sources()


def versions():
    return {k: importlib.metadata.version(k) for k in ("torch", "numpy", "clip")}
