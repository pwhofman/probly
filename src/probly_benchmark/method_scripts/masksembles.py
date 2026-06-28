"""Masksembles benchmark code."""

from __future__ import annotations

import logging

import torch

from probly.quantification import quantify
from probly.representer import Sampler
from probly.transformation.masksembles import masksembles
from probly_benchmark.models import LeNet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = LeNet(n_classes=10)
cep = masksembles(model, predictor_type="logit_classifier")
print(cep)
rep = Sampler(cep, num_samples=10)
logger.info(rep)
inputs = torch.randn(3, 1, 28, 28)
logger.info(cep(inputs))
output = rep.predict(inputs)
logger.info(output)
logger.info(output.shape)
quantification = quantify(output)
logger.info(quantification)
logger.info(quantification.total)  # ty:ignore[unresolved-attribute]
