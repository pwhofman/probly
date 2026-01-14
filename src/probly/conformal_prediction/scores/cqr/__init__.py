"""Conformal Prediction CQR score implementation."""

from probly.conformal_prediction.scores.cqr.common import CQRScore, cqr_score_func
from probly.lazy_types import JAX_ARRAY


# Lazy registration - these will only be imported when needed
@cqr_score_func.delayed_register(JAX_ARRAY)
def _(_: type) -> None:
    from probly.conformal_prediction.scores.cqr import flax  # noqa: PLC0415, F401


__all__ = ["CQRScore", "cqr_score_func"]
