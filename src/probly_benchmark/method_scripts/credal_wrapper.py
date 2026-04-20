"""Benchmarking for the credal wrapper method."""

from __future__ import annotations

import logging

import torch

from probly.method.credal_wrapper import credal_wrapper
from probly.quantification import quantify
from probly.representer import representer
from probly_benchmark.models import LeNet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = LeNet(n_classes=5)
cep = credal_wrapper(model, num_members=10, predictor_type="probabilistic_classifier")
rep = representer(cep)
logger.info(rep)
inputs = torch.randn(3, 1, 28, 28)
output = rep.predict(inputs)
logger.info(output)
logger.info(output.shape)
quantification = quantify(output)
logger.info(quantification)
logger.info(quantification.total)  # ty:ignore[unresolved-attribute]
