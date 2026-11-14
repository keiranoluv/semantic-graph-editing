"""Loss and graph decoding with a separate per-input dynamic label namespace."""
import torch

from tools.edit_model_v1.codec import score_predictions
from tools.magicbrush_experiment_v3.decoding import decode_one
from tools.magicbrush_experiment_v3.targets import encode_targets, structured_loss

from .candidates import candidates
from .features import collate_inputs


def targets_for(examples, vocab, config, *, use_copy, device="cpu"):
    targets = encode_targets(examples, vocab, config, device)
    if not use_copy:
        return targets
    for b, e in enumerate(examples):
        spans = candidates(e.instruction)
        source = {n.id: i for i, n in enumerate(e.source.nodes)}
        added = sorted(n.id for n in e.target.nodes if n.id not in source)
        for op in e.operations:
            if op.kind not in ("add_node", "replace_node"):
                continue
            prefix = "new_node" if op.kind == "add_node" else "node"
            slot = added.index(op.entity_id) if prefix == "new_node" else source[op.entity_id]
            # Existing COPY and train vocabulary retain their ordinary indices.
            if targets[prefix + "_label"][b, slot] == -100 and op.after.label in spans:
                targets[prefix + "_label"][b, slot] = 1 + len(vocab["labels"]) + spans.index(op.after.label)
    return targets


def decode_batch(items, logits, vocab, max_nodes, *, use_copy):
    logits = {k: v.detach().cpu() for k, v in logits.items()}
    rows = []
    for b, item in enumerate(items):
        local = {**vocab, "labels": vocab["labels"] + list(candidates(item["instruction"]))} if use_copy else vocab
        # Batch padding and merged duplicates are masked by the model.
        rows.append(decode_one(item, logits, local, max_nodes, b))
    return rows


def training_step(model, optimizer, examples, store, config, device):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    inputs = collate_inputs([e.model_input() for e in examples], store, device=device)
    targets = targets_for(examples, model.vocabulary, config, use_copy=model.use_copy, device=device)
    loss = structured_loss(model(inputs), targets, config)
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite loss")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach()), len(examples)


def validation_loss(model, examples, store, config, device):
    model.eval()
    total = 0.0
    with torch.inference_mode():
        for start in range(0, len(examples), config["val_batch_size"]):
            es = examples[start : start + config["val_batch_size"]]
            inputs = collate_inputs([e.model_input() for e in es], store, device=device)
            targets = targets_for(es, model.vocabulary, config, use_copy=model.use_copy, device=device)
            loss = structured_loss(model(inputs), targets, config)
            total += float(loss) * len(es)
    return total / len(examples)


def validate(model, examples, store, config, device):
    model.eval()
    rows = []
    with torch.inference_mode():
        for start in range(0, len(examples), config["val_batch_size"]):
            es = examples[start : start + config["val_batch_size"]]
            items = [e.model_input() for e in es]
            logits = model(collate_inputs(items, store, device=device))
            rows.extend(decode_batch(items, logits, model.vocabulary, model.max_nodes, use_copy=model.use_copy))
    metrics, errors = score_predictions(examples, [r["tokens"] for r in rows])
    _, raw_errors = score_predictions(examples, [r["raw_tokens"] for r in rows])
    return metrics, [
        {"id": e.id, **r, "error": error, "raw_error": raw_error}
        for e, r, error, raw_error in zip(examples, rows, errors, raw_errors, strict=True)
    ]
