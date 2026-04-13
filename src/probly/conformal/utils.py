"""Utility functions for conformal predictors."""

from __future__ import annotations

from typing import TYPE_CHECKING

from probly.traverse_nn import nn_compose

if TYPE_CHECKING:
    from probly.calibrator._common import ConformalCalibrator
from pytraverse import GlobalVariable, function_traverser, lazydispatch_traverser
from pytraverse.core import State, traverse_with_state

calibration_traverser = lazydispatch_traverser(name="_check_calibrated")

P = GlobalVariable("is_calibrated")


@calibration_traverser.register(object)
def _(obj: object, state: State) -> tuple[object, State]:
    score_attribute = getattr(obj, "non_conformity_score", None)
    quantile_attribute = getattr(obj, "conformal_quantile", None)
    if score_attribute is not None and quantile_attribute is not None:
        state[P] = True
    return obj, state


def is_conformal_calibrated(predictor: ConformalCalibrator) -> bool:
    """Check if the conformal predictor is calibrated."""
    score_attribute = getattr(predictor, "non_conformity_score", None)
    quantile_attribute = getattr(predictor, "conformal_quantile", None)
    if score_attribute is not None and quantile_attribute is not None:
        return True
    _, state = traverse_with_state(
        predictor,
        nn_compose(calibration_traverser, function_traverser),
        init={P: False},
    )
    return state[P]
