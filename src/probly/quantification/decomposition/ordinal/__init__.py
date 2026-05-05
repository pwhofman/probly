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
    categorical_variance_aleatoric,
    categorical_variance_total,
    ordinal_variance_of_expected_predictive_distribution,
)
from probly.quantification.measure.variance import conditional_variance, mutual_information_variance

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
    "conditional_variance",
    "mutual_information_variance",
    "labelwise_conditional_entropy",
    "labelwise_entropy_of_expected_predictive_distribution",
    "labelwise_conditional_variance",
    "labelwise_variance_of_expected_predictive_distribution",
    "ordinal_conditional_entropy",
    "ordinal_entropy_of_expected_predictive_distribution",
    "ordinal_conditional_variance",
    "ordinal_variance_of_expected_predictive_distribution",
]
