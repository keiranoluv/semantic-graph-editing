"""Versioned, CPU-only EMNLP 2020 graph supervision adapter."""

from common.io import sha256

from common.paths import ROOT

VERSION = "emnlp2020-graph-modification-v1"

ARCHIVE_SHA256 = "6e13b096f2756b638f59e38b13b80658f91f53cbfa6750a210929fa8a091d337"

OFFICIAL_COMMIT = "6e1b396007a0c938bccda2bbddade332745b3f42"

ARCHIVE = ROOT / "data/raw/emnlp2020-graph-modification-v1/scene_graph_modification_data.tgz"

POLICY = "A_preserve_topology_reciprocal_undirected_v1"

ALIGNMENT = "exhaustive_unique_minimum_maximum_cardinality_v1"


def verified_archive(path):
    actual = sha256(path)
    if actual != ARCHIVE_SHA256:
        raise ValueError("official_archive_checksum_mismatch")
    return actual
