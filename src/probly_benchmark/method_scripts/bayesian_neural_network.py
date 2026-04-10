"""Dropout benchmark code."""

from __future__ import annotations

import logging

import torch

from probly.method.bayesian import bayesian
from probly.representer import representer
from probly_benchmark.models import LeNet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = LeNet(n_classes=5)
cep = bayesian(model)
rep = representer(cep, num_samples=10)
logger.info(rep)
inputs = torch.randn(3, 1, 28, 28)
output = rep.predict(inputs)
logger.info(output)
logger.info(output.shape)
