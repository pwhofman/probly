"""Benchmark for Mahalanobis out-of-distribution detection."""

from __future__ import annotations

import logging

import torch

from probly.method.mahalanobis import mahalanobis
from probly.quantification import quantify
from probly.representer import representer
from probly_benchmark.models import MiniResNet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = MiniResNet(n_classes=5)
cep = mahalanobis(model, predictor_type="logit_classifier")

# Mahalanobis is post-hoc: the class-conditional Gaussians must be fitted on
# training features before scoring, and the multi-layer combiner is optionally
# calibrated on in- vs out-of-distribution inputs.
train_inputs = torch.randn(32, 1, 28, 28)
train_labels = torch.randint(0, 5, (32,))
cep.fit_mahalanobis_heads(train_inputs, train_labels)
ood_inputs = torch.randn(32, 1, 28, 28) * 5.0 + 20.0
cep.fit_combiner(train_inputs, ood_inputs)

rep = representer(cep)
logger.info(rep)
inputs = torch.randn(3, 1, 28, 28)
output = rep.predict(inputs)
logger.info(output)
logger.info(output.shape)
quantification = quantify(output)
logger.info(quantification)
logger.info(quantification.epistemic)  # ty:ignore[unresolved-attribute]
