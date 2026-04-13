"""Conformal predictors for classification using scikit-learn."""

from __future__ import annotations

import copy

from sklearn.base import ClassifierMixin

from ._common import conformal_generator


@conformal_generator.register(ClassifierMixin)
def _(model: ClassifierMixin) -> ClassifierMixin:
    conformal_model = copy.deepcopy(model)
    conformal_model.conformal_quantile = None  # ty: ignore[unresolved-attribute]
    conformal_model.non_conformity_score = None  # ty: ignore[unresolved-attribute]
    return conformal_model
