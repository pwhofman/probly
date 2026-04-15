"""Benchmark for credal ensembling."""

from __future__ import annotations

import logging

import torch

from probly.method.ddu import ddu
from probly.representer import representer
from probly_benchmark.models import MiniResNet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = MiniResNet(n_classes=5)
cep = ddu(model, predictor_type="logit_classifier")
rep = representer(cep)
logger.info(rep)
inputs = torch.randn(3, 1, 28, 28)
output = rep.predict(inputs)
logger.info(output)
logger.info(output.shape)
