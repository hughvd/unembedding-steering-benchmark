"""Residual-stream hook shared by the layer/scale grid searches."""

import torch


@torch.no_grad()
def extract_layer_features(model, tokenizer, text, layer, sae=None):
    """Pool non-padding tokens after decoder block `layer`, optionally through an SAE."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
    hidden = model(**inputs, output_hidden_states=True).hidden_states[layer + 1]
    features = sae.encode(hidden) if sae is not None else hidden
    mask = inputs["attention_mask"].unsqueeze(-1)
    pooled = (features * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    return pooled.squeeze(0).float().cpu()


def decode_sae_direction(direction, decoder):
    """Map an additive latent intervention into residual coordinates."""
    decoder = decoder.detach().to(device=direction.device, dtype=direction.dtype)
    if direction.ndim != 1 or decoder.ndim != 2 or direction.shape[0] != decoder.shape[0]:
        raise ValueError("Expected a latent vector and a [latent, residual] decoder")
    return direction @ decoder


def steer_residual(output, direction, scale):
    """Add a unit direction to every token, scaled per example's last-token norm.

    Transformers versions return either a tensor or a tuple from a decoder
    layer. Preserve the original structure and any attention/cache outputs.
    """
    if scale == 0:
        return output
    hidden = output[0] if isinstance(output, tuple) else output
    vector = direction.to(device=hidden.device, dtype=hidden.dtype).reshape(-1)
    if vector.numel() != hidden.shape[-1]:
        raise ValueError("Steering direction must be in residual coordinates; decode SAE directions first")
    norm = vector.float().norm()
    if not torch.isfinite(norm) or norm == 0:
        raise ValueError("Steering direction must have a finite nonzero norm")
    vector = vector / norm.to(hidden.dtype)
    magnitude = hidden[:, -1:, :].float().norm(dim=-1, keepdim=True).to(hidden.dtype)
    steered = hidden + scale * magnitude * vector
    return (steered, *output[1:]) if isinstance(output, tuple) else steered
