"""Matched graph encoders with categorical or compositional attribute output."""
from tools.magicbrush_easy_copy_v1.model import StructuredPredictor as NameCopyModel

from .sequence import AttributeDecoder


class StructuredPredictor(NameCopyModel):
    def __init__(self, *, sequence_attributes=False, attribute_tokens=None, token_vectors=None, **kwargs):
        super().__init__(**kwargs)
        self.sequence_attributes = sequence_attributes
        self._contexts = []
        self.attributes.register_forward_pre_hook(self._capture_context)
        # Allocate after base modules: common parameters retain seed parity.
        self.attribute_decoder = (
            AttributeDecoder(kwargs.get("hidden", 192), attribute_tokens, token_vectors)
            if sequence_attributes
            else None
        )

    def _capture_context(self, module, args):
        self._contexts.append(args[0])

    def forward(self, batch):
        self._contexts = []
        result = super().forward(batch)
        values = self._contexts
        self._contexts = []
        if self.sequence_attributes:
            for key in list(result):
                if key.endswith(("_attributes", "_attribute_count")):
                    del result[key]
            result["_node_context"], result["_new_node_context"] = values
        return result
