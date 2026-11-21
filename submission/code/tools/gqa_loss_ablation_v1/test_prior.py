import torch
from torch_geometric.data import Data

from src.training.gqa_pilot import collate
from tools.gqa_loss_ablation_v1.__main__ import training_prior
from tools.run_gqa_full_prepared import prepare_supervision


def test_prior_matches_sampled_soft_targets_including_empty_supervision():
    def graph(relations, predicates):
        pairs = torch.combinations(torch.arange(8), r=2)
        candidates = torch.cat([pairs, pairs.flip([1])]).T.contiguous()
        data = Data(
            x=torch.zeros(8, 2),
            object_labels=torch.ones(8, dtype=torch.long),
            edge_index=candidates,
            edge_label_index=candidates,
            gt_relation_index=torch.tensor(relations).long().reshape(2, -1),
            gt_predicate=torch.tensor(predicates, dtype=torch.long),
        )
        return prepare_supervision(data, 4)

    graphs = [graph([[0, 0, 0], [1, 1, 1]], [1, 2, 1]), graph([[], []], [])]
    sampled, _ = collate(graphs, 4, torch.Generator().manual_seed(123))
    prior = training_prior(graphs, 4)
    assert prior["target_mass"] == sampled["targets"].sum(dim=0).tolist() == [35, 0.5, 0.5, 0]
    assert prior["prior"] == [36 / 40, 1.5 / 40, 1.5 / 40, 1 / 40]
