"""Deterministic, group-distinct train cohort and frozen v3 feature reuse."""
import random
from collections import Counter, defaultdict

import torch

from src.editing.operations import KINDS
from tools.edit_model_v1.model import collate_inputs
from tools.edit_model_v1.text import load_store
from tools.magicbrush_experiment_v3.cache import input_texts
from tools.magicbrush_experiment_v3.vocabulary import prototype_inputs
from tools.prepare_magicbrush import digest


def input_key(e):
    return digest({"source": e.source.to_dict(), "instruction": e.instruction})


def select_train(train, size, seed):
    if not train or any(e.split != "train" for e in train):
        raise ValueError("Only train examples may enter the memorization cohort")
    outcomes = defaultdict(set)
    for e in train:
        outcomes[input_key(e)].add(digest(e.target.to_dict()))
    conflicts = {k for k, v in outcomes.items() if len(v) > 1}
    pool = sorted((e for e in train if input_key(e) not in conflicts), key=lambda e: e.id)
    random.Random(seed).shuffle(pool)
    frequencies = Counter(op.kind for e in train for op in e.operations)
    chosen, groups, inputs, covered = [], set(), set(), set()

    def eligible(e):
        return e.group not in groups and input_key(e) not in inputs and e not in chosen

    def append(e):
        chosen.append(e)
        groups.add(e.group)
        inputs.add(input_key(e))
        covered.update(op.kind for op in e.operations)

    for kind in sorted(KINDS, key=lambda k: (frequencies[k], k)):
        if kind in covered:
            continue
        e = next((e for e in pool if eligible(e) and any(o.kind == kind for o in e.operations)), None)
        if e is None:
            raise ValueError("Cannot cover operation type with distinct, nonconflicting train inputs: " + kind)
        append(e)
    for e in pool:
        if len(chosen) >= size:
            break
        if eligible(e):
            append(e)
    if len(chosen) != size:
        raise ValueError("Cannot form requested distinct-group cohort")
    return sorted(chosen, key=lambda e: e.id), {
        "policy": "seeded shuffled train list; cover rare operation kinds first, fill to size; distinct source groups and semantic inputs; no performance-based selection",
        "selection_seed": seed,
        "conflicting_input_keys_excluded": len(conflicts),
        "full_train_operation_support": dict(frequencies),
        "cohort_operation_support": dict(Counter(o.kind for e in chosen for o in e.operations)),
        "distinct_groups": len(groups),
    }


def example_receipt(e):
    return {
        "id": e.id,
        "split": e.split,
        "source_group": e.group,
        "instruction": e.instruction,
        "input_sha256": input_key(e),
        "target_sha256": digest(e.target.to_dict()),
        "operations_sha256": digest([o.to_dict() for o in e.operations]),
        "operation_types": sorted({o.kind for o in e.operations}),
    }


def reuse_store(directory, parent_identity, cohort, vocab):
    store, meta = load_store(directory, parent_identity)
    # Materialize only the selected train inputs + full TRAIN vocabulary into the
    # probe's in-memory store. No validation vectors are accessed for learning.
    from tools.edit_model_v1.text import TextStore, text_key

    texts = input_texts(list(cohort) + prototype_inputs(vocab))
    vectors = torch.stack([store[t] for t in texts])
    selected = TextStore([text_key(t) for t in texts], vectors)
    batch = collate_inputs([e.model_input() for e in cohort], selected)
    signatures = []
    for i in range(len(cohort)):
        import hashlib

        h = hashlib.sha256()
        for name, value in sorted(batch.items()):
            h.update(name.encode())
            h.update(value[i].contiguous().numpy().tobytes())
        signatures.append(h.hexdigest())
    if len(set(signatures)) != len(signatures):
        raise ValueError("Selected examples collide in actual frozen model input tensors")
    return selected, meta, signatures
