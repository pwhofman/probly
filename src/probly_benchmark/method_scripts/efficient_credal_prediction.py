"""Efficient credal prediction benchmark code."""

from __future__ import annotations

import logging

import torch

from probly.method.efficient_credal_prediction import efficient_credal_prediction
from probly.representer import representer
from probly_benchmark.models import LeNet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = LeNet(n_classes=5)
cep = efficient_credal_prediction(model, num_classes=5, predictor_type="logit_classifier")
rep = representer(cep)
logger.info(rep)
inputs = torch.randn(3, 1, 28, 28)
output = rep.predict(inputs)
logger.info(output)
logger.info(output.shape)
