"""Conformal predictors for classification using scikit-learn."""
from __future__ import annotations

import copy

from sklearn.base import ClassifierMixin

from ._common import conformal_generator, to_probabilities


@conformal_generator.register(ClassifierMixin)
def _(model: ClassifierMixin) -> ClassifierMixin:
    conformal_model = copy.deepcopy(model)
    conformal_model.conformal_quantile = None
    conformal_model.non_conformity_score = None
    return conformal_model

