"""PyTorch implementation of the selective prediction metrics."""

from __future__ import annotations

import torch

from ._common import (
    augrc,
    aurc,
    check_coverage,
    check_inputs,
    check_no_nan,
    check_risk,
    coverage_at_risk,
    risk_at_coverage,
    risk_coverage_curve,
)


@risk_coverage_curve.register(torch.Tensor)
def torch_risk_coverage_curve(
    criterion: torch.Tensor, losses: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the exact risk-coverage curve for PyTorch tensors."""
    losses = torch.as_tensor(losses, device=criterion.device)
    n = check_inputs(criterion, losses)
    check_no_nan(torch.isnan(criterion).any())
    # Half precision counts exactly only up to 2048 and overflows past 65504, so the curve is computed in at
    # least float32.
    if losses.is_floating_point():
        losses = losses.to(torch.promote_types(losses.dtype, torch.float32))
    else:
        losses = losses.to(torch.get_default_dtype())
    order = torch.argsort(criterion, stable=True)
    criterion_sorted = criterion[order]
    losses_sorted = losses[order]

    # Tied instances are accepted together: every position in a run of tied criterion values points to the end
    # of that run, as in probly.metrics.torch._binary_clf_curve. Ties are detected in the dtype of the
    # criterion, since a cast to the dtype of the losses could merge distinct values.
    end = torch.ones(1, dtype=torch.bool, device=criterion.device)
    is_run_end = torch.cat([criterion_sorted[1:] != criterion_sorted[:-1], end])
    run_end = torch.where(is_run_end, torch.arange(n, device=criterion.device), n - 1)
    run_end = torch.cummin(run_end.flip(0), dim=0).values.flip(0)

    count = (run_end + 1).to(losses.dtype)
    risk = torch.cumsum(losses_sorted, dim=0)[run_end] / count
    coverage = torch.cat([torch.zeros(1, dtype=losses.dtype, device=losses.device), count / n])
    risk = torch.cat([risk[:1], risk])
    threshold_dtype = criterion.dtype if criterion.is_floating_point() else losses.dtype
    start = torch.full((1,), -torch.inf, dtype=threshold_dtype, device=criterion.device)
    thresholds = torch.cat([start, criterion_sorted.to(threshold_dtype)])
    return coverage, risk, thresholds


@aurc.register(torch.Tensor)
def torch_aurc(criterion: torch.Tensor, losses: torch.Tensor) -> torch.Tensor:
    """Compute the area under the exact risk-coverage curve for PyTorch tensors."""
    coverage, risk, _ = torch_risk_coverage_curve(criterion, losses)
    return torch.trapezoid(risk, coverage)


@augrc.register(torch.Tensor)
def torch_augrc(criterion: torch.Tensor, losses: torch.Tensor) -> torch.Tensor:
    """Compute the area under the exact generalized risk-coverage curve for PyTorch tensors."""
    coverage, risk, _ = torch_risk_coverage_curve(criterion, losses)
    return torch.trapezoid(risk * coverage, coverage)


@risk_at_coverage.register(torch.Tensor)
def torch_risk_at_coverage(criterion: torch.Tensor, losses: torch.Tensor, coverage: float) -> torch.Tensor:
    """Compute the selective risk at a target coverage for PyTorch tensors."""
    check_coverage(coverage)
    curve_coverage, risk, _ = torch_risk_coverage_curve(criterion, losses)
    # The coverage is non-decreasing, so the first point that reaches the target has the smallest coverage.
    return risk[torch.argmax((curve_coverage >= coverage).to(torch.uint8))]


@coverage_at_risk.register(torch.Tensor)
def torch_coverage_at_risk(criterion: torch.Tensor, losses: torch.Tensor, risk: float) -> torch.Tensor:
    """Compute the largest coverage with at most a target selective risk for PyTorch tensors."""
    check_risk(risk)
    coverage, curve_risk, _ = torch_risk_coverage_curve(criterion, losses)
    # The endpoint at coverage 0 is excluded, since it accepts no instance.
    return torch.where(curve_risk[1:] <= risk, coverage[1:], 0.0).max()
