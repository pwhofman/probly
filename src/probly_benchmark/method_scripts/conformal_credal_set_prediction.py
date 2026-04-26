"""Conformal credal set prediction benchmark code."""

from __future__ import annotations

import logging
from typing import cast

import numpy as np
from scipy.stats import entropy as scipy_entropy
import torch

from probly.calibrator._common import calibrate
from probly.conformal_scores.total_variation import TVScore
from probly.method.conformal_credal_set_prediction._common import (
    ConformalCredalSetPredictor,
    conformal_credal_set_prediction,
)
from probly.predictor import predict
from probly.quantification import QuantificationResult, quantify
from probly.quantification.measure.credal_set import lower_entropy, upper_entropy
from probly.representation.credal_set.array import ArrayDistanceBasedCredalSet
from probly.representer import Sampler
from probly_benchmark.models import LeNet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@upper_entropy.register(ArrayDistanceBasedCredalSet)
def array_distance_based_upper_entropy(credal_set: ArrayDistanceBasedCredalSet, base: str | None = None) -> float:
    """Pragmatic upper entropy: entropy of nominal + radius."""
    p = credal_set.nominal.probabilities
    return scipy_entropy(p, axis=-1, base=base) + credal_set.radius


@lower_entropy.register(ArrayDistanceBasedCredalSet)
def array_distance_based_lower_entropy(credal_set: ArrayDistanceBasedCredalSet, base: str | None = None) -> float:
    """Pragmatic lower entropy: max(0, entropy of nominal - radius)."""
    p = credal_set.nominal.probabilities
    return np.clip(scipy_entropy(p, axis=-1, base=base) - credal_set.radius, 0, None)


model = LeNet(n_classes=5)
model.eval()
cep = conformal_credal_set_prediction(model, k_classes=5)
logger.info(cep)
x_calib = torch.randn(10, 1, 28, 28)
y_calib = torch.randint(0, 5, (10,))
alpha = 0.1
with torch.no_grad():
    calibrated_cep = cast("ConformalCredalSetPredictor", calibrate(cep, TVScore(), x_calib, y_calib, alpha))
logger.info(calibrated_cep)
logger.info(calibrated_cep.quantile)
rep = Sampler(calibrated_cep, num_samples=10)
logger.info(rep)

inputs = torch.randn(3, 1, 28, 28)
with torch.no_grad():
    prediction = predict(calibrated_cep, inputs)
    credal_quantification: QuantificationResult = quantify(prediction)
    logger.info(credal_quantification)
    samples = rep.predict(inputs)
    logger.info(samples)
    logger.info(samples.shape)
