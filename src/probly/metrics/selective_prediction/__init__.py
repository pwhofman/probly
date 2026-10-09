"""Selective prediction metrics with backend dispatch for NumPy, PyTorch, and JAX.

A selective predictor abstains on the instances with the largest criterion, typically an uncertainty score, and
:func:`risk_coverage_curve` traces the resulting trade-off between coverage and selective risk. To compare criteria,
the curve is summarized by an area: :func:`augrc` is the recommended summary, while the more common :func:`aurc`
overweights failures at low coverage. Either area depends on the mean loss as well as on the ranking, so the risk at
full coverage is worth reporting alongside it. A deployed predictor, in contrast, runs at a single working point,
which :func:`risk_at_coverage` and :func:`coverage_at_risk` report.
"""

from __future__ import annotations

from probly.lazy_types import JAX_ARRAY, JAX_ARRAY_LIKE, JAX_TRACER, TORCH_TENSOR, TORCH_TENSOR_LIKE

from . import numpy as numpy
from ._common import augrc, aurc, coverage_at_risk, risk_at_coverage, risk_coverage_curve


@augrc.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@aurc.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@coverage_at_risk.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@risk_at_coverage.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@risk_coverage_curve.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


@augrc.delayed_register((JAX_ARRAY, JAX_ARRAY_LIKE, JAX_TRACER))
@aurc.delayed_register((JAX_ARRAY, JAX_ARRAY_LIKE, JAX_TRACER))
@coverage_at_risk.delayed_register((JAX_ARRAY, JAX_ARRAY_LIKE, JAX_TRACER))
@risk_at_coverage.delayed_register((JAX_ARRAY, JAX_ARRAY_LIKE, JAX_TRACER))
@risk_coverage_curve.delayed_register((JAX_ARRAY, JAX_ARRAY_LIKE, JAX_TRACER))
def _(_: type) -> None:
    from . import jax as jax  # noqa: PLC0415


__all__ = [
    "augrc",
    "aurc",
    "coverage_at_risk",
    "risk_at_coverage",
    "risk_coverage_curve",
]
