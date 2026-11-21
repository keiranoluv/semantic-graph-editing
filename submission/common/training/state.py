"""Stable model-state fingerprint for matched GQA experiments."""

import hashlib


def state_hash(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(str((tuple(value.shape), str(value.dtype))).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
