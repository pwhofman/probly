"""Total Variation nonconformity score."""

from __future__ import annotations

from probly.lazy_types import TORCH_TENSOR, TORCH_TENSOR_LIKE

from ._common import TVScore, tv_score_func


@tv_score_func.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = ["TVScore", "tv_score_func"]
