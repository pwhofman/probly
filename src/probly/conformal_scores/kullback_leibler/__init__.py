"""Kullback-Leibler divergence nonconformity score."""

from __future__ import annotations

from probly.lazy_types import TORCH_TENSOR, TORCH_TENSOR_LIKE

from ._common import KLDivergenceScore, kl_divergence_score, kl_divergence_score_func


@kl_divergence_score_func.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = ["KLDivergenceScore", "kl_divergence_score", "kl_divergence_score_func"]
