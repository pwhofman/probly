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


def _torch_sorted_runs(
    criterion: torch.Tensor, losses: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort by the criterion and find the runs of tied criterion values.

    Returns:
        The sorted criterion, the sorted losses (in at least float32), and for every position the first and the
        last position of its run.
    """
    losses = torch.as_tensor(losses, device=criterion.device)
    n = check_inputs(criterion, losses)
    check_no_nan(torch.isnan(criterion).any())
    # Half precision counts exactly only up to 2048 and overflows past 65504, so the curve is computed in at
    # least float32.
    if losses.is_floating_point():
        losses = losses.to(torch.promote_types(losses.dtype, torch.float32))
    else:
        losses = losses.to(torch.get_default_dtype())
    check_no_nan(torch.isnan(losses).any(), "losses")
    order = torch.argsort(criterion, stable=True)
    criterion_sorted = criterion[order]
    losses_sorted = losses[order]

    # Ties are detected in the dtype of the criterion, since a cast to the dtype of the losses could merge
    # distinct values.
    is_new = criterion_sorted[1:] != criterion_sorted[:-1]
    true = torch.ones(1, dtype=torch.bool, device=criterion.device)
    positions = torch.arange(n, device=criterion.device)
    run_start = torch.cummax(torch.where(torch.cat([true, is_new]), positions, 0), dim=0).values
    run_end = torch.where(torch.cat([is_new, true]), positions, n - 1)
    run_end = torch.cummin(run_end.flip(0), dim=0).values.flip(0)
    return criterion_sorted, losses_sorted, run_start, run_end


@risk_coverage_curve.register(torch.Tensor)
def torch_risk_coverage_curve(
    criterion: torch.Tensor, losses: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the exact risk-coverage curve for PyTorch tensors."""
    criterion_sorted, losses_sorted, _, run_end = _torch_sorted_runs(criterion, losses)
    n = len(losses_sorted)

    # Tied instances are accepted together: every position in a run of tied criterion values points to the end
    # of that run, as in probly.metrics.torch._binary_clf_curve.
    count = (run_end + 1).to(losses_sorted.dtype)
    risk = torch.cumsum(losses_sorted, dim=0)[run_end] / count
    coverage = torch.cat([torch.zeros(1, dtype=risk.dtype, device=risk.device), count / n])
    risk = torch.cat([risk[:1], risk])
    threshold_dtype = criterion.dtype if criterion.is_floating_point() else risk.dtype
    start = torch.full((1,), -torch.inf, dtype=threshold_dtype, device=criterion.device)
    thresholds = torch.cat([start, criterion_sorted.to(threshold_dtype)])
    return coverage, risk, thresholds


@aurc.register(torch.Tensor)
def torch_aurc(criterion: torch.Tensor, losses: torch.Tensor) -> torch.Tensor:
    """Compute the area under the exact risk-coverage curve for PyTorch tensors."""
    _, losses_sorted, run_start, run_end = _torch_sorted_runs(criterion, losses)
    n = len(losses_sorted)

    # Accepting the k most confident instances, with a random part of a tied run, gives an expected cumulative
    # loss that is linear in k inside the run. The expected selective risk at every k is that loss over k.
    cumulative = torch.cumsum(losses_sorted, dim=0)
    before_run = torch.where(run_start > 0, cumulative[run_start - 1], 0.0)
    count = (torch.arange(n, device=run_end.device) + 1).to(cumulative.dtype)
    run_length = (run_end - run_start + 1).to(cumulative.dtype)
    expected_loss = before_run + (cumulative[run_end] - before_run) * (count - run_start) / run_length
    risk = expected_loss / count
    coverage = torch.cat([torch.zeros(1, dtype=risk.dtype, device=risk.device), count / n])
    return torch.trapezoid(torch.cat([risk[:1], risk]), coverage)


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
