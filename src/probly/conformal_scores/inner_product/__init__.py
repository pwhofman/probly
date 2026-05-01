"""Inner Product nonconformity score."""

from __future__ import annotations

from probly.lazy_types import TORCH_TENSOR, TORCH_TENSOR_LIKE

from ._common import InnerProductScore, inner_product_score, inner_product_score_func


@inner_product_score_func.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = ["InnerProductScore", "inner_product_score", "inner_product_score_func"]
