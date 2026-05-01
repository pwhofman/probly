"""Torch implementation of the efficient credal prediction method."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import nn

from ._common import (
    _validate_alpha,
    compute_efficient_credal_bounds,
    compute_efficient_credal_prediction_bounds,
    efficient_credal_prediction_generator,
)


@efficient_credal_prediction_generator.register(nn.Module)
class TorchEfficientCredalPredictor(nn.Module):
    """Torch nn.Module that wraps a softmax-free model and stores credal bounds."""

    def __init__(self, predictor: nn.Module) -> None:
        """Initialize the predictor.

        Args:
            predictor: The base model.
        """
        super().__init__()
        self.predictor = predictor
        self.register_buffer("lower", None)
        self.register_buffer("upper", None)

    @property
    def lower_bounds(self) -> torch.Tensor:
        """Per-class lower probability bounds."""
        return cast("torch.Tensor", self.lower)

    @property
    def upper_bounds(self) -> torch.Tensor:
        """Per-class upper probability bounds."""
        return cast("torch.Tensor", self.upper)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the predictor."""
        return self.predictor(x)


@compute_efficient_credal_prediction_bounds.register(torch.Tensor)
def _compute_bounds_torch(
    logits_train: torch.Tensor,
    targets_train: torch.Tensor,
    num_classes: int,
    alpha: float,
    *,
    chunk_size: int = 128,
    n_iter: int = 50,
    bracket: float = 1e4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-class additive logit bounds via classwise relative-likelihood optimization.

    For each class ``k`` and each direction (min/max), find the optimal
    additive logit perturbation ``x`` on column ``k`` that keeps the mean
    training relative likelihood at least ``alpha``. The relative
    likelihood is ``exp(ll(logits + x * e_k) - ll(logits))`` where ``ll``
    is the mean per-sample log-likelihood.

    Based on :cite:`hofmanefficient` and the reference implementation at
    https://github.com/pwhofman/efficient-credal-prediction/blob/main/models.py.
    Solved by parallel bisection across all classes on the GPU after an
    analytical rewrite of the constraint.
    """
    _validate_alpha(alpha)
    with torch.no_grad():
        device = logits_train.device
        n_samples = logits_train.shape[0]
        logits_f64 = logits_train.detach().to(dtype=torch.float64)
        log_p = torch.nn.functional.log_softmax(logits_f64, dim=1)

        # Stable log(1 - p)
        log_1mp = torch.where(
            log_p > -math.log(2),
            torch.log(-torch.expm1(log_p)),
            torch.log1p(-torch.exp(log_p)),
        )
        threshold = math.log(alpha) if alpha > 0 else -float("inf")
        class_counts = torch.bincount(targets_train.detach().to(device).long(), minlength=num_classes).to(torch.float64)
        freq = class_counts / n_samples

        lower = torch.zeros(num_classes, dtype=torch.float64, device=device)
        upper = torch.zeros(num_classes, dtype=torch.float64, device=device)

        for start in range(0, num_classes, chunk_size):
            end = min(start + chunk_size, num_classes)
            log_p_chunk = log_p[:, start:end]
            log_1mp_chunk = log_1mp[:, start:end]
            freq_chunk = freq[start:end]
            chunk_width = end - start

            # Upper bound: bisect on x in [0, bracket].
            left = torch.zeros(chunk_width, dtype=torch.float64, device=device)
            right = torch.full((chunk_width,), bracket, dtype=torch.float64, device=device)
            for _ in range(n_iter):
                mid = (left + right) / 2.0
                f_k = mid * freq_chunk - torch.mean(torch.logaddexp(log_1mp_chunk, log_p_chunk + mid), dim=0)
                feasible = f_k > threshold
                left = torch.where(feasible, mid, left)
                right = torch.where(feasible, right, mid)
            upper[start:end] = (left + right) / 2.0

            # Lower bound: bisect on x in [-bracket, 0].
            left = torch.full((chunk_width,), -bracket, dtype=torch.float64, device=device)
            right = torch.zeros(chunk_width, dtype=torch.float64, device=device)
            for _ in range(n_iter):
                mid = (left + right) / 2.0
                f_k = mid * freq_chunk - torch.mean(torch.logaddexp(log_1mp_chunk, log_p_chunk + mid), dim=0)
                feasible = f_k > threshold
                right = torch.where(feasible, mid, right)
                left = torch.where(feasible, left, mid)
            lower[start:end] = (left + right) / 2.0

    return lower, upper


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
