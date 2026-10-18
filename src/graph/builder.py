"""Scene graph conversion contract, including leakage-safe topology."""

import torch
from torch_geometric.data import Data

from src.datasets.schemas import SceneGraphRecord


class SceneGraphBuilder:
    """Create a PyG Data object from annotation and node features.

    edge_index: message-passing topology, independent of annotated relations.
    edge_label_index: directed candidate object pairs for predicate prediction.
    edge_label: IDs aligned to candidate pairs (background=0).

    Multi-label annotations on one pair require an explicit policy before
    supervised target construction. Do not silently drop duplicate pairs.
    """

    def build(self, record: SceneGraphRecord, x: torch.Tensor) -> Data:
        """Return Data with x, edge_index, edge_label_index, edge_label,
        boxes, object_labels and image_id.

        Required invariant: row i in x corresponds to record.objects[i].
        """
        raise NotImplementedError("PyG graph materialization belongs to Phase 4")
