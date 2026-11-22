"""Structured losses and inference; scoring remains the frozen operation metric."""

import torch
from task2_graph_editing.magicbrush.sequence_model.codec import score_predictions
from task2_graph_editing.magicbrush.sequence_model.model import collate_inputs
from task2_graph_editing.magicbrush.structured_model.decoding import decode_batch
from task2_graph_editing.magicbrush.structured_model.targets import encode_targets, structured_loss


def training_step(model, optimizer, examples, store, config, device):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    inputs = collate_inputs([e.model_input() for e in examples], store, device=device)
    targets = encode_targets(examples, model.vocabulary, config, device)
    loss = structured_loss(model(inputs), targets, config)
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite structured training loss")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach()), len(examples)


def validation_loss(model, examples, store, config, device):
    model.eval()
    total = 0.0
    with torch.inference_mode():
        for start in range(0, len(examples), config["val_batch_size"]):
            batch = examples[start : start + config["val_batch_size"]]
            inputs = collate_inputs([e.model_input() for e in batch], store, device=device)
            targets = encode_targets(batch, model.vocabulary, config, device)
            loss = structured_loss(model(inputs), targets, config)
            total += float(loss) * len(batch)
    return total / len(examples)


def validate(model, examples, store, config, device):
    model.eval()
    rows = []
    with torch.inference_mode():
        for start in range(0, len(examples), config["val_batch_size"]):
            batch = examples[start : start + config["val_batch_size"]]
            # This is the only data passed to inference: no targets/operations.
            items = [e.model_input() for e in batch]
            inputs = collate_inputs(items, store, device=device)
            rows.extend(decode_batch(items, model(inputs), model.vocabulary, model.max_nodes))
    metrics, errors = score_predictions(examples, [r["tokens"] for r in rows])
    _, raw_errors = score_predictions(examples, [r["raw_tokens"] for r in rows])
    return metrics, [
        {"id": e.id, **row, "error": error, "raw_error": raw_error}
        for e, row, error, raw_error in zip(examples, rows, errors, raw_errors, strict=True)
    ]
