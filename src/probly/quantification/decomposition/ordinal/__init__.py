"""Ordinal decomposition methods."""

from probly.quantification.decomposition.variance._common import CategoricalVarianceDecomposition
from probly.quantification.measure.ordinal import (
    categorical_variance_aleatoric,
    categorical_variance_total,
    labelwise_conditional_entropy,
    labelwise_entropy_of_expected_predictive_distribution,
    labelwise_expected_conditional_variance,
    labelwise_variance_of_expected_predictive_distribution,
    ordinal_conditional_entropy,
    ordinal_entropy_of_expected_predictive_distribution,
    ordinal_expected_conditional_variance,
    ordinal_variance_of_expected_predictive_distribution,
)
from probly.quantification.measure.variance import expected_conditional_variance, variance_of_conditional_mean

from ._common import (
    LabelwiseBinaryEntropyDecomposition,
    LabelwiseBinaryVarianceDecomposition,
    OrdinalEntropyDecomposition,
    OrdinalVarianceDecomposition,
)

__all__ = [
    "CategoricalVarianceDecomposition",
    "LabelwiseBinaryEntropyDecomposition",
    "LabelwiseBinaryVarianceDecomposition",
    "OrdinalEntropyDecomposition",
    "OrdinalVarianceDecomposition",
    "categorical_variance_aleatoric",
    "categorical_variance_total",
    "expected_conditional_variance",
    "labelwise_conditional_entropy",
    "labelwise_entropy_of_expected_predictive_distribution",
    "labelwise_expected_conditional_variance",
    "labelwise_variance_of_expected_predictive_distribution",
    "ordinal_conditional_entropy",
    "ordinal_entropy_of_expected_predictive_distribution",
    "ordinal_expected_conditional_variance",
    "ordinal_variance_of_expected_predictive_distribution",
    "variance_of_conditional_mean",
]
