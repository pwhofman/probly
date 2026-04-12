"""Conformal regression methods for scikit-learn models."""

from __future__ import annotations

import copy

from sklearn.base import RegressorMixin

from ._common import conformal_generator


@conformal_generator.register(RegressorMixin)
def _(model: RegressorMixin) -> RegressorMixin:
    conformal_model = copy.deepcopy(model)
    conformal_model.conformal_quantile = None
    conformal_model.non_conformity_score = None

    return conformal_model
