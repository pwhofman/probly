"""Ordinal classification and regression uncertainty decompositions."""

from probly.lazy_types import TORCH_TENSOR, TORCH_TENSOR_LIKE

from . import array as _array  # noqa: F401  (registers numpy implementations)
from ._common import (
    CategoricalVarianceDecomposition,
    GaussianVarianceDecomposition,
    LabelwiseBinaryEntropyDecomposition,
    LabelwiseBinaryVarianceDecomposition,
    OrdinalEntropyDecomposition,
    OrdinalVarianceDecomposition,
    categorical_variance_aleatoric,
    categorical_variance_total,
    gaussian_variance_aleatoric,
    gaussian_variance_epistemic,
    labelwise_binary_entropy_aleatoric,
    labelwise_binary_entropy_total,
    labelwise_binary_variance_aleatoric,
    labelwise_binary_variance_total,
    ordinal_binary_entropy_aleatoric,
    ordinal_binary_entropy_total,
    ordinal_binary_variance_aleatoric,
    ordinal_binary_variance_total,
)


@ordinal_binary_entropy_total.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@ordinal_binary_entropy_aleatoric.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@ordinal_binary_variance_total.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@ordinal_binary_variance_aleatoric.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@labelwise_binary_entropy_total.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@labelwise_binary_entropy_aleatoric.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@labelwise_binary_variance_total.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@labelwise_binary_variance_aleatoric.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@categorical_variance_total.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@categorical_variance_aleatoric.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@gaussian_variance_aleatoric.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@gaussian_variance_epistemic.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "CategoricalVarianceDecomposition",
    "GaussianVarianceDecomposition",
    "LabelwiseBinaryEntropyDecomposition",
    "LabelwiseBinaryVarianceDecomposition",
    "OrdinalEntropyDecomposition",
    "OrdinalVarianceDecomposition",
    "categorical_variance_aleatoric",
    "categorical_variance_total",
    "gaussian_variance_aleatoric",
    "gaussian_variance_epistemic",
    "labelwise_binary_entropy_aleatoric",
    "labelwise_binary_entropy_total",
    "labelwise_binary_variance_aleatoric",
    "labelwise_binary_variance_total",
    "ordinal_binary_entropy_aleatoric",
    "ordinal_binary_entropy_total",
    "ordinal_binary_variance_aleatoric",
    "ordinal_binary_variance_total",
]
