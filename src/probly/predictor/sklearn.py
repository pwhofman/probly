"""sklearn-specific predictor dispatch helpers."""

from __future__ import annotations

from sklearn.base import BaseEstimator

from probly.representation.distribution import CategoricalDistribution, create_categorical_distribution

from ._common import LogitDistributionPredictor, predict_categorical_distribution_from_logit, predict_raw


@predict_categorical_distribution_from_logit.register(BaseEstimator)
def sklearn_logit_prediction_to_distribution[**In, Out: CategoricalDistribution](
    predictor: LogitDistributionPredictor[In, Out], *args: In.args, **kwargs: In.kwargs
) -> CategoricalDistribution:
    """Interpret sklearn logit-predictor outputs as probabilities."""
    return create_categorical_distribution(predict_raw(predictor, *args, **kwargs))
