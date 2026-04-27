"""Quantification of Deep Deterministic Uncertainty (DDU) representations."""

from __future__ import annotations

from probly.lazy_types import TORCH_TENSOR_LIKE

from ._common import DDUDecomposition, ddu_epistemic_uncertainty


@ddu_epistemic_uncertainty.delayed_register(TORCH_TENSOR_LIKE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = ["DDUDecomposition", "ddu_epistemic_uncertainty"]
