"""CPU graph topology and collation; no GT relation labels and no PyG dependency."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Graph:
    video_id: str
    # Metadata columns: original frame index, original track ID, object class ID.
    # Track IDs are keys, never numeric model features.
    nodes: np.ndarray
    spatial_edges: np.ndarray
    temporal_edges: np.ndarray
    self_edges: np.ndarray
    query_edges: np.ndarray


def batch_graphs(graphs):
    """Offset every edge set, including query edges, by sample-local node counts.

    Two windows from the same video remain separate graph samples. Frame/track
    metadata retains original IDs; sample indices disambiguate repeated IDs.
    """
    graphs = list(graphs)
    fields = ("spatial_edges", "temporal_edges", "self_edges", "query_edges")
    pointers = [0]
    node_parts, batch_parts = [], []
    edge_parts = {name: [] for name in fields}
    for sample, graph in enumerate(graphs):
        if graph.nodes.ndim != 2 or graph.nodes.shape[1] != 3 or graph.nodes.dtype.kind not in "iu":
            raise ValueError("Graph nodes must be an integer [N,3] metadata array")
        n = len(graph.nodes)
        for name in fields:
            edges = getattr(graph, name)
            if edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype.kind not in "iu":
                raise ValueError(f"Invalid edge array: {name}")
            if edges.size and (edges.min() < 0 or edges.max() >= n):
                raise ValueError(f"Edge index outside sample nodes: {name}")
            edge_parts[name].append(edges + pointers[-1])
        node_parts.append(graph.nodes)
        batch_parts.append(np.full(n, sample, dtype=np.int64))
        pointers.append(pointers[-1] + n)
    result = {
        "video_ids": [graph.video_id for graph in graphs],
        "node_ptr": np.asarray(pointers, dtype=np.int64),
        "nodes": np.concatenate(node_parts) if node_parts else np.empty((0, 3), dtype=np.int64),
        "node_batch": np.concatenate(batch_parts) if batch_parts else np.empty(0, dtype=np.int64),
    }
    for name in fields:
        result[name] = np.concatenate(edge_parts[name], axis=1) if graphs else np.empty((2, 0), dtype=np.int64)
    query = result["query_edges"]
    result["query_batch"] = result["node_batch"][query[0]]
    result["query_frame_indices"] = result["nodes"][query[0], 0]
    result["query_track_pairs"] = result["nodes"][query, 1].T.copy()
    result["temporal_delta_frames"] = (
        result["nodes"][result["temporal_edges"][1], 0] - result["nodes"][result["temporal_edges"][0], 0]
    )
    return result
