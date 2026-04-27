"""Active learning query strategies with backend dispatch for NumPy and PyTorch."""

from probly.lazy_types import TORCH_TENSOR

from . import array as array
from ._common import (
    BadgeEstimator,
    BADGEQuery,
    Estimator,
    MarginSampling,
    QueryStrategy,
    RandomQuery,
    UncertaintyQuery,
    badge_select,
    margin_select,
    uncertainty_select,
)


@margin_select.delayed_register(TORCH_TENSOR)
@uncertainty_select.delayed_register(TORCH_TENSOR)
@badge_select.delayed_register(TORCH_TENSOR)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "BADGEQuery",
    "BadgeEstimator",
    "Estimator",
    "MarginSampling",
    "QueryStrategy",
    "RandomQuery",
    "UncertaintyQuery",
    "badge_select",
    "margin_select",
    "uncertainty_select",
]
