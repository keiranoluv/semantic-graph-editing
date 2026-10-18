"""Scene graph conversion contract, including leakage-safe topology."""

from dataclasses import dataclass
from typing import Literal

import torch
from torch_geometric.data import Data

from src.datasets.schemas import SceneGraphRecord

Topology = Literal["fully_connected"]
CandidateMode = Literal["all_pairs", "supervised_pairs"]


@dataclass(frozen=True)
class GraphBuilderConfig:
    """Configuration for scene graph construction."""

    topology: Topology = "fully_connected"
    candidate_mode: CandidateMode = "all_pairs"
    include_self_loops: bool = False


class SceneGraphBuilder:
    """Create a PyG Data object from annotation and node features.

    edge_index: message-passing topology, independent of annotated relations.
    edge_label_index: directed candidate object pairs for predicate prediction.
    edge_label: IDs aligned to candidate pairs (background=0).

    Multi-label annotations on one pair require an explicit policy before
    supervised target construction. Do not silently drop duplicate pairs.
    """

    def __init__(self, config: GraphBuilderConfig | None = None) -> None:
        self.config = config if config is not None else GraphBuilderConfig()

    def build(self, record: SceneGraphRecord, x: torch.Tensor) -> Data:
        """Return Data with x, edge_index, edge_label_index, edge_label,
        boxes, object_labels and image_id.

        Required invariant: row i in x corresponds to record.objects[i].
        """
        raise NotImplementedError("PyG graph materialization belongs to Phase 4")
