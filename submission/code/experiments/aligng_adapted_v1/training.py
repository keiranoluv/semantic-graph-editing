"""Loss/optimizer wiring only; P0 intentionally has no GPU or experiment launcher."""

import torch

from src.training.penet import prototype_losses

from .batching import model_inputs


def batch_loss(model, batch, device="cpu"):
    output = model(**model_inputs(batch, device), return_embeddings=True)
    # Preserve PE-Net k1=10, k2=1, gamma1=1, gamma2=7 and unit weights.
    return prototype_losses(*output, batch["targets"].to(device), mode=model.config.loss_mode)


def train_step(model, batch, optimizer, device="cpu", *, gradient_clip_norm=5.0):
    """One explicit update; no update (including AdamW decay/momentum) on empty queries."""
    optimizer.zero_grad(set_to_none=True)
    model.train()
    losses = batch_loss(model, batch, device)
    if not len(batch["targets"]):
        return {"losses": losses, "updated": False, "gradient_norm": None}
    if not torch.isfinite(losses["total"]):
        raise RuntimeError("Non-finite loss")
    losses["total"].backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm, error_if_nonfinite=True)
    optimizer.step()
    return {"losses": losses, "updated": True, "gradient_norm": float(norm)}
