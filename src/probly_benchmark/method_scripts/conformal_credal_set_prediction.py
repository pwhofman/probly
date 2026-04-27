"""Conformal credal set prediction benchmark code."""

from __future__ import annotations

import logging

import torch

from probly.calibrator._common import calibrate
from probly.conformal_scores.total_variation import TVScore
from probly.method.conformal_credal_set_prediction._common import (
    conformal_credal_set_prediction,
)
from probly.predictor import predict
from probly.quantification import QuantificationResult, quantify
from probly.representer import representer
from probly_benchmark.models import LeNet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


model = LeNet(n_classes=5)
model.eval()
cep = conformal_credal_set_prediction(model, k_classes=5)
logger.info(cep)
inputs = torch.randn(3, 1, 28, 28)
x_calib = torch.randn(10, 1, 28, 28)
y_calib = torch.randint(0, 5, (10,))
alpha = 0.1
calibrated_cep = calibrate(cep, TVScore(), x_calib, y_calib, alpha)
logger.info(calibrated_cep)
logger.info(calibrated_cep.quantile)
rep = representer(calibrated_cep)
logger.info(rep)
prediction = predict(rep, inputs)
credal_quantification: QuantificationResult = quantify(prediction)
logger.info(credal_quantification)
