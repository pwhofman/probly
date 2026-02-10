"""Conformal prediction module imports and structure."""

from probly.conformal_prediction.methods.class_conditional import ClassConditionalClassifier, ClassConditionalRegressor
from probly.conformal_prediction.methods.common import ConformalClassifier, ConformalPredictor, ConformalRegressor
from probly.conformal_prediction.methods.cvplus import CVPlusClassifier, CVPlusRegressor
from probly.conformal_prediction.methods.jackknifeplus import (
    JackknifeCVBase,
    JackknifePlusClassifier,
    JackknifePlusRegressor,
)
from probly.conformal_prediction.methods.mondrian import (
    GroupedConformalBase,
    MondrianConformalClassifier,
    MondrianConformalRegressor,
)
from probly.conformal_prediction.methods.split import SplitConformal, SplitConformalClassifier, SplitConformalRegressor
from probly.conformal_prediction.scores.absolute_error.common import AbsoluteErrorScore
from probly.conformal_prediction.scores.aps.common import APSScore
from probly.conformal_prediction.scores.common import ClassificationScore, RegressionScore, Score
from probly.conformal_prediction.scores.cqr.common import CQRScore
from probly.conformal_prediction.scores.lac.common import LACScore
from probly.conformal_prediction.scores.raps.common import RAPSScore
from probly.conformal_prediction.scores.saps.common import SAPSScore
from probly.conformal_prediction.utils.metrics import average_set_size, empirical_coverage

__all__ = [
    "APSScore",
    "AbsoluteErrorScore",
    "CQRScore",
    "CVPlusClassifier",
    "CVPlusRegressor",
    "ClassConditionalClassifier",
    "ClassConditionalRegressor",
    "ClassificationScore",
    "ConformalClassifier",
    "ConformalPredictor",
    "ConformalRegressor",
    "GroupedConformalBase",
    "JackknifeCVBase",
    "JackknifePlusClassifier",
    "JackknifePlusRegressor",
    "LACScore",
    "MondrianConformalClassifier",
    "MondrianConformalRegressor",
    "RAPSScore",
    "RegressionScore",
    "SAPSScore",
    "Score",
    "SplitConformal",
    "SplitConformalClassifier",
    "SplitConformalRegressor",
    "average_set_size",
    "empirical_coverage",
]
