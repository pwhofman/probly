"""Benchmarking for the credal relative likelihood method."""

from __future__ import annotations

import logging

import torch

from probly.method.credal_relative_likelihood import credal_relative_likelihood
from probly.representer import CredalRelativeLikelihoodRepresenter
from probly_benchmark.models import LeNet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = LeNet(n_classes=5)
cep = credal_relative_likelihood(model, num_members=10, predictor_type="probabilistic_classifier")
rep = CredalRelativeLikelihoodRepresenter(cep)
logger.info(rep)
inputs = torch.randn(3, 1, 28, 28)
output = rep.predict(inputs)
logger.info(output)
logger.info(output.shape)
