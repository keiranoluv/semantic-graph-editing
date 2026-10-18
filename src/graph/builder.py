"""Build PyG scene graphs with label-independent message passing."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

import torch
from torch_geometric.data import Data

from src.datasets.schemas import SceneGraphRecord

Topology = Literal["fully_connected"]
CandidateMode = Literal["all_pairs", "supervised_pairs"]


@dataclass(frozen=True)
class GraphBuilderConfig:
    """Configuration of graph topology and predicate candidates."""

    topology: Topology = "fully_connected"
    candidate_mode: CandidateMode = "all_pairs"
    include_self_loops: bool = False


class SceneGraphBuilder:
    """Convert normalized annotations and features into PyG Data.

    Message-passing topology never depends on ground-truth relations.
    Original predicate annotations are preserved separately.
    """

    def __init__(self, config: GraphBuilderConfig | None = None) -> None:
        self.config = config if config is not None else GraphBuilderConfig()

    def build(self, record: SceneGraphRecord, x: torch.Tensor) -> Data:
        """Build a graph from one image and its ordered node features."""
        record.validate()

        if self.config.topology != "fully_connected":
            raise NotImplementedError("Only fully_connected is supported")

        if self.config.candidate_mode != "all_pairs":
            raise NotImplementedError("Only all_pairs is supported")

        num_nodes = len(record.objects)

        if x.ndim != 2 or x.shape[0] != num_nodes:
            raise ValueError("Expected node features with shape [N, D]")

        if x.dtype != torch.float32:
            raise ValueError("Node features must use torch.float32")

        # Map original instance identifiers to contiguous node indices.
        id_to_index = {obj.object_id: index for index, obj in enumerate(record.objects)}

        # Message passing and prediction both use all ordered pairs.
        pairs = [
            (src, dst)
            for src in range(num_nodes)
            for dst in range(num_nodes)
            if self.config.include_self_loops or src != dst
        ]

        if pairs:
            edge_index = torch.tensor(pairs, dtype=torch.long).T.contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        # Preserve every annotated relationship, including duplicates.
        gt_pairs = []
        gt_predicates = []
        labels_by_pair = defaultdict(list)

        for relation in record.relations:
            src = id_to_index[relation.subject_id]
            dst = id_to_index[relation.object_id]

            gt_pairs.append((src, dst))
            gt_predicates.append(relation.predicate_id)
            labels_by_pair[(src, dst)].append(relation.predicate_id)

        if gt_pairs:
            gt_relation_index = torch.tensor(gt_pairs, dtype=torch.long).T.contiguous()
        else:
            gt_relation_index = torch.empty((2, 0), dtype=torch.long)

        # Single-label projection for a first-stage classifier.
        # -1 means this pair has multiple distinct predicate labels.
        edge_labels = []

        for pair in pairs:
            predicates = set(labels_by_pair.get(pair, []))

            if not predicates:
                edge_labels.append(0)
            elif len(predicates) == 1:
                edge_labels.append(next(iter(predicates)))
            else:
                edge_labels.append(-1)

        boxes = torch.tensor(
            [obj.bbox_xyxy for obj in record.objects],
            dtype=torch.float32,
        ).reshape(num_nodes, 4)

        object_labels = torch.tensor(
            [obj.label_id for obj in record.objects],
            dtype=torch.long,
        )

        return Data(
            x=x,
            edge_index=edge_index,
            edge_label_index=edge_index.clone(),
            edge_label=torch.tensor(edge_labels, dtype=torch.long),
            boxes=boxes,
            object_labels=object_labels,
            object_ids=torch.tensor(
                [obj.object_id for obj in record.objects],
                dtype=torch.long,
            ),
            gt_relation_index=gt_relation_index,
            gt_predicate=torch.tensor(gt_predicates, dtype=torch.long),
            image_id=record.image_id,
            num_nodes=num_nodes,
        )
