"""Conformal credal set prediction benchmark code."""

from __future__ import annotations

import logging

import torch


from probly.conformal_scores.total_variation import TVScore
from probly.method.conformal_credal_set_prediction._common import (
    conformal_total_variation,
)
from probly.predictor import predict
from probly.quantification import QuantificationResult, quantify
from probly.representer import representer
from probly_benchmark.models import LeNet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


model = LeNet(n_classes=5)
model.eval()
cep = conformal_total_variation(model)
logger.info(cep)
inputs = torch.randn(3, 1, 28, 28)
x_calib = torch.randn(10, 1, 28, 28)
y_calib = torch.randint(0, 5, (10,))
alpha = 0.1

#def calibrate(self, alpha: float, y_calib: Out, *calib_args: In.args, **calib_kwargs: In.kwargs) -> Self:
calibrated_cep = cep.calibrate(alpha, y_calib, x_calib)
logger.info(cep)
logger.info(cep.quantile)
rep = representer(cep)
logger.info(rep)
prediction = rep.predict(inputs)
credal_quantification: QuantificationResult = quantify(prediction)
logger.info(credal_quantification)
