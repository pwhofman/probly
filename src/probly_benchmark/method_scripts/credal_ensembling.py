"""Benchmark for credal ensembling."""

from __future__ import annotations

import logging

import torch

from probly.method.credal_ensembling import credal_ensembling
from probly.representer import representer
from probly_benchmark.models import LeNet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = LeNet(n_classes=5)
cep = credal_ensembling(model, num_members=10, predictor_type="probabilistic_classifier")
rep = representer(cep, alpha=0.1, distance="euclidean")
logger.info(rep)
inputs = torch.randn(3, 1, 28, 28)
output = rep.predict(inputs)
logger.info(output)
logger.info(output.shape)
