"""Torch implementation of the efficient credal prediction representer."""

from __future__ import annotations

import torch

from probly.representer.efficient_credal_prediction._common import compute_efficient_credal_bounds


@compute_efficient_credal_bounds.register(torch.Tensor)
def _(logits: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Compute the packed credal interval bounds via 2K logit perturbations.

    Builds 2K perturbed logit tensors per input by broadcasting (no in-place
    mutation across iterations -- diverging from the authors' reference
    implementation which mutates the logit buffer and accumulates drift).
    """
    n_classes = logits.shape[-1]
    eye = torch.eye(n_classes, device=logits.device, dtype=logits.dtype)
    perturb_up = upper.unsqueeze(-1) * eye
    perturb_lo = lower.unsqueeze(-1) * eye

    # 2K perturbed logit tensors per input. Broadcasting produces a fresh
    # (B, K, K) tensor for each side; original logits are not modified.
    logits_up = logits.unsqueeze(1) + perturb_up.unsqueeze(0)
    logits_lo = logits.unsqueeze(1) + perturb_lo.unsqueeze(0)

    probs_up = torch.softmax(logits_up, dim=-1)
    probs_lo = torch.softmax(logits_lo, dim=-1)

    # Per-output-class min/max across all 2K perturbations.
    probs_all = torch.cat([probs_up, probs_lo], dim=1)
    lower_bounds = probs_all.min(dim=1).values
    upper_bounds = probs_all.max(dim=1).values

    return torch.cat([lower_bounds, upper_bounds], dim=-1)
