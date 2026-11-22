"""Train/validation loader and explicit human-review provenance gate. No test labels."""

import json
from dataclasses import dataclass

from common.editing.operations import EditGraph


@dataclass(frozen=True)
class Example:
    id: str
    split: str
    group: str
    instruction: str
    source: EditGraph
    target: EditGraph
    operations: tuple

    @property
    def simple(self):
        # Predeclared graph-level proxy, not a claim about linguistic simplicity.
        return len(self.operations) == 1

    def model_input(self):
        return {"source": self.source, "instruction": self.instruction}


def feature_texts(source, instruction):
    # Deliberately accepts neither Example nor target/operation metadata.
    return {
        "instruction": instruction,
        "nodes": [json.dumps([n.label, list(n.attributes)], ensure_ascii=False) for n in source.nodes],
        "edges": [e.predicate for e in source.edges],
    }
