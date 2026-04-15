"""Benchmarking of the DARE method."""

from __future__ import annotations

import logging

import torch

from probly.method.dare import dare
from probly.representer import representer
from probly_benchmark.models import LeNet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = LeNet(n_classes=5)
dare_ensemble = dare(model, num_members=5, lambda_reg=0.01, threshold=0.5)
rep = representer(dare_ensemble)
logger.info(rep)
inputs = torch.randn(3, 1, 28, 28)
output = rep.predict(inputs)
logger.info(output)
logger.info(output.shape)
