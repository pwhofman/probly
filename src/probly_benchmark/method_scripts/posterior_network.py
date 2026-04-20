"""Posterior network benchmark code."""

from __future__ import annotations

import logging

import torch
from torch import nn

from probly.method.posterior_network import posterior_network
from probly.representer import representer

LATENT_DIM = 16

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5),  # (1,28,28) -> (6,24,24)
    nn.Tanh(),
    nn.AvgPool2d(2),  # -> (6,12,12)
    nn.Conv2d(6, 16, kernel_size=5),  # -> (16,8,8)
    nn.Tanh(),
    nn.AvgPool2d(2),  # -> (16,4,4)
    nn.Flatten(),
    nn.Linear(16 * 4 * 4, 120),
    nn.Tanh(),
    nn.Linear(120, 84),
    nn.Tanh(),
    nn.Linear(84, LATENT_DIM),
)
cep = posterior_network(
    model, latent_dim=LATENT_DIM, num_classes=3, class_counts=[1, 2, 0], predictor_type="probabilistic_classifier"
)
rep = representer(cep)
logger.info(rep)
inputs = torch.randn(3, 1, 28, 28)
output = rep.predict(inputs)
logger.info(output)
logger.info(output.concentration)
