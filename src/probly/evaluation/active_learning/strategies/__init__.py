"""Active learning query strategies with backend dispatch for NumPy and PyTorch."""

from probly.lazy_types import TORCH_TENSOR

from . import array as array
from ._badge import BADGEQuery, badge_embed, badge_select
from ._protocols import BadgeEstimator, Estimator, QueryStrategy, UncertaintyEstimator
from ._scores import entropy_score, least_confident_score, margin_score
from ._selection import random_select, topk_select
from ._strategies import (
    EntropySampling,
    LeastConfidentSampling,
    MarginSampling,
    RandomQuery,
    UncertaintyQuery,
)


@least_confident_score.delayed_register(TORCH_TENSOR)
@margin_score.delayed_register(TORCH_TENSOR)
@topk_select.delayed_register(TORCH_TENSOR)
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
    "badge_embed",
    "badge_select",
    "entropy_score",
    "least_confident_score",
    "margin_score",
    "random_select",
    "topk_select",
]
