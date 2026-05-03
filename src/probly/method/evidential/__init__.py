"""Evidential methods for probly."""

from probly.method.evidential.classification import EvidentialClassificationPredictor, evidential_classification
from probly.method.evidential.regression import evidential_regression

__all__ = ["EvidentialClassificationPredictor", "evidential_classification", "evidential_regression"]
