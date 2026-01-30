"""Absolute error score module."""

from probly.conformal_prediction.scores.absolute_error.common import AbsoluteErrorScore, absolute_error_score_func
from probly.lazy_types import JAX_ARRAY, TORCH_TENSOR


# Lazy registration - these will only be imported when needed
@absolute_error_score_func.delayed_register(TORCH_TENSOR)
def _(_: type) -> None:
    from probly.conformal_prediction.scores.absolute_error import torch  # noqa: PLC0415,F401


@absolute_error_score_func.delayed_register(JAX_ARRAY)
def _(_: type) -> None:
    from probly.conformal_prediction.scores.absolute_error import flax  # noqa: PLC0415,F401


__all__ = ["AbsoluteErrorScore", "absolute_error_score_func"]
