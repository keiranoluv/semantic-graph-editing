"""The v2 encoder with device-correct padding mask; v2 source stays frozen."""

import torch

from task2_graph_editing.he2020.data_processing.model import LexicalPredictor as CPULexicalPredictor


class LexicalPredictor(CPULexicalPredictor):
    def encode(self, batch):
        # Same frozen encoder computation, with padding sized to the current
        # split rather than paying projection cost for 42 absent edges.
        instruction = self.text_projection(batch["instruction"])
        nodes = self.text_projection(batch["nodes"]) + self.node_identity.weight[: batch["nodes"].shape[1]]
        edges = self.text_projection(batch["edges"]) + self.edge_identity.weight[: batch["edges"].shape[1]]
        nodes = nodes * batch["node_mask"].unsqueeze(-1)
        for layer in self.layers:
            nodes = layer(nodes, edges, batch)
        subject = batch["subject"].unsqueeze(-1).expand(-1, -1, nodes.shape[-1])
        obj = batch["object"].unsqueeze(-1).expand(-1, -1, nodes.shape[-1])
        edges = self.edge_projection(torch.cat((nodes.gather(1, subject), edges, nodes.gather(1, obj)), -1))
        memory = torch.cat((instruction.unsqueeze(1), nodes, edges), 1)
        mask = torch.cat(
            (torch.ones(len(nodes), 1, dtype=torch.bool, device=nodes.device), batch["node_mask"], batch["edge_mask"]),
            1,
        )
        pooled = (memory * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        initial = torch.tanh(self.initial(torch.cat((pooled, instruction), -1)))
        return memory, mask, pooled, initial.unsqueeze(0)
