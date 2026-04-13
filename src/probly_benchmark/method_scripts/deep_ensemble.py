"""Benchmarking of the deep ensemble method."""

from __future__ import annotations

import logging

import torch

from probly.method.ensemble import ensemble
from probly.representer import representer
from probly_benchmark.models import LeNet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = LeNet(n_classes=5)
cep = ensemble(model, num_members=10)
rep = representer(cep)
logger.info(rep)
inputs = torch.randn(3, 1, 28, 28)
output = rep.predict(inputs)
logger.info(output)
logger.info(output.shape)
