"""Conformal Prediction APS score implementation."""

from probly.conformal_prediction.scores.aps.common import APSScore, aps_score_func
from probly.lazy_types import JAX_ARRAY, TORCH_TENSOR


# Lazy registration - these will only be imported when needed
@aps_score_func.delayed_register(TORCH_TENSOR)
def _(_: type) -> None:
    from probly.conformal_prediction.scores.aps import torch  # noqa: PLC0415,F401


@aps_score_func.delayed_register(JAX_ARRAY)
def _(_: type) -> None:
    from probly.conformal_prediction.scores.aps import flax  # noqa: PLC0415,F401


__all__ = ["APSScore", "aps_score_func"]
