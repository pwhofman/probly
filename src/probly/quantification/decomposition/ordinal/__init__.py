"""Ordinal decomposition methods."""

from probly.quantification.decomposition.variance._common import CategoricalVarianceDecomposition
from probly.quantification.measure.ordinal import (
    labelwise_conditional_entropy,
    labelwise_conditional_variance,
    labelwise_entropy_of_expected_predictive_distribution,
    labelwise_variance_of_expected_predictive_distribution,
    ordinal_conditional_entropy,
    ordinal_conditional_variance,
    ordinal_entropy_of_expected_predictive_distribution,
    ordinal_integer_variance_aleatoric,
    ordinal_integer_variance_total,
    ordinal_variance_of_expected_predictive_distribution,
)
from probly.quantification.measure.variance import conditional_variance, mutual_information

from ._common import (
    LabelwiseBinaryEntropyDecomposition,
    LabelwiseBinaryVarianceDecomposition,
    OrdinalEntropyDecomposition,
    OrdinalVarianceDecomposition,
)

# Standalone function aliases using the old naming convention
ordinal_binary_entropy_total = ordinal_entropy_of_expected_predictive_distribution
ordinal_binary_entropy_aleatoric = ordinal_conditional_entropy
ordinal_binary_variance_total = ordinal_variance_of_expected_predictive_distribution
ordinal_binary_variance_aleatoric = ordinal_conditional_variance
labelwise_binary_entropy_total = labelwise_entropy_of_expected_predictive_distribution
labelwise_binary_entropy_aleatoric = labelwise_conditional_entropy
labelwise_binary_variance_total = labelwise_variance_of_expected_predictive_distribution
labelwise_binary_variance_aleatoric = labelwise_conditional_variance
categorical_variance_total = ordinal_integer_variance_total
categorical_variance_aleatoric = ordinal_integer_variance_aleatoric
gaussian_variance_aleatoric = conditional_variance
gaussian_variance_epistemic = mutual_information

__all__ = [
    "CategoricalVarianceDecomposition",
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
