"""Masksembles benchmark code."""

from __future__ import annotations

import logging

import torch

from probly.quantification import quantify
from probly.representer import representer
from probly.transformation.masksembles import masksembles
from probly_benchmark.models import SimpleCNN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = SimpleCNN(n_classes=10)
cep = masksembles(model, predictor_type="logit_classifier")
rep = representer(cep)
logger.info(rep)
inputs = torch.randn(3, 1, 28, 28)
logger.info(cep(inputs))
with torch.no_grad():
    output = rep.represent(inputs)
logger.info(output)
logger.info(output.tensor.probabilities.shape)
quantification = quantify(output)
logger.info(quantification)
logger.info(quantification.total)  # ty:ignore[unresolved-attribute]
