"""Benchmarking for the credal bnn method."""

from __future__ import annotations

import logging

import torch

from probly.method.credal_bnn import credal_bnn
from probly.representer import CredalBNNRepresenter
from probly_benchmark.models import LeNet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = LeNet(n_classes=5)
cep = credal_bnn(model, num_members=10)
rep = CredalBNNRepresenter(cep)
logger.info(rep)
inputs = torch.randn(3, 1, 28, 28)
output = rep.predict(inputs)
logger.info(output)
logger.info(output.shape)
