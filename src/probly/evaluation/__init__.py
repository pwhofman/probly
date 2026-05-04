"""Evaluation metrics for predicted-set representations.

Public entry points:

* :func:`coverage` -- empirical coverage of a predicted set.
* :func:`efficiency` -- average size of a predicted set.
* :func:`average_interval_width` -- mean per-class interval width for
  envelope-based credal sets.

NumPy-backed implementations are loaded eagerly. PyTorch handlers are loaded
lazily on first dispatch involving any torch-like type, mirroring the pattern
used by :mod:`probly.metrics`.
"""

from __future__ import annotations

from probly.evaluation import array as array
from probly.evaluation.metrics import average_interval_width, coverage, efficiency
from probly.lazy_types import TORCH_TENSOR, TORCH_TENSOR_LIKE


@average_interval_width.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@coverage.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@efficiency.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
def _(_: type) -> None:
    from probly.evaluation import torch as torch  # noqa: PLC0415


__all__ = [
    "average_interval_width",
    "coverage",
    "efficiency",
]
