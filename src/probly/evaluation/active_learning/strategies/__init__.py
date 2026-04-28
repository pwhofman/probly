"""Active learning query strategies with backend dispatch for NumPy and PyTorch."""

from probly.lazy_types import TORCH_TENSOR

from . import array as array
from ._common import (
    BadgeEstimator,
    BADGEQuery,
    EntropySampling,
    Estimator,
    LeastConfidentSampling,
    MarginSampling,
    QueryStrategy,
    RandomQuery,
    UncertaintyEstimator,
    UncertaintyQuery,
    badge_select,
    entropy_select,
    least_confident_select,
    margin_select,
    random_select,
    uncertainty_select,
)


@entropy_select.delayed_register(TORCH_TENSOR)
@least_confident_select.delayed_register(TORCH_TENSOR)
@margin_select.delayed_register(TORCH_TENSOR)
@uncertainty_select.delayed_register(TORCH_TENSOR)
@badge_select.delayed_register(TORCH_TENSOR)
@random_select.delayed_register(TORCH_TENSOR)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "BADGEQuery",
    "BadgeEstimator",
    "EntropySampling",
    "Estimator",
    "LeastConfidentSampling",
    "MarginSampling",
    "QueryStrategy",
    "RandomQuery",
    "UncertaintyEstimator",
    "UncertaintyQuery",
    "badge_select",
    "entropy_select",
    "least_confident_select",
    "margin_select",
    "random_select",
    "uncertainty_select",
]
